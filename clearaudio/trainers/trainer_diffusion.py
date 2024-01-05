import copy
import datetime
import functools
import logging
import os
from pathlib import Path

import blobfile as bf
from clearaudio.datasets.audio import create_dataloader, create_generator_from_dataloader, AudioDatasetExtended
from clearaudio.models.guided_diffusion.script_util import args_to_dict, args_to_dict_if_present, create_model_and_diffusion, model_and_diffusion_defaults
from clearaudio.trainers.base_trainer import BaseTrainer
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from omegaconf import DictConfig, OmegaConf, open_dict

import torchaudio

from torch.utils.tensorboard import SummaryWriter

from ..models.guided_diffusion import dist_util, logger
from ..models.guided_diffusion.fp16_util import MixedPrecisionTrainer
from ..models.guided_diffusion.nn import update_ema
from ..models.guided_diffusion.resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


LOG = logging.getLogger(__name__)

class DiffusionTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def setup_trainer(self) -> None:
    #     self,
    #     *,
    #     model,
    #     diffusion,
    #     data,
    #     batch_size,
    #     microbatch,
    #     lr,
    #     ema_rate,
    #     log_interval,
    #     save_interval,
    #     resume_checkpoint,
    #     use_fp16=False,
    #     fp16_scale_growth=1e-3,
    #     schedule_sampler=None,
    #     weight_decay=0.0,
    #     lr_anneal_steps=0,
    # ):
        LOG.info(f"DiffusionTrainer: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}")
        torchaudio.set_audio_backend("sox_io")
        if self.cfg.trainer.platform == "slurm":
            torchaudio.set_audio_backend('soundfile')
        self.writer = None
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if self.cfg.trainer.rank == 0:
            if self.cfg.trainer.use_clearml:
                from clearml import Task
                task = Task.init(project_name="ClearAudio", task_name=self.cfg.trainer.ml_exp_name)
            if self.cfg.trainer.use_wandb:
                import wandb
                wandb.init(project="ClearAudio", entity=self.cfg.trainer.wandb_entity, sync_tensorboard=True)
                wandb.run.name = self.cfg.trainer.ml_exp_name
                wandb.run.save()
            self.writer = SummaryWriter()
        self.sample_rate = self.cfg.dataset.sample_rate


        logdir = str(Path(self.cfg.trainer.log_dir) / datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
        logger.configure(dir=logdir)

        LOG.info("creating model and diffusion...")

        # update default values using the config
        model_and_diffusion_args = model_and_diffusion_defaults()
        model_and_diffusion_args.update(
            args_to_dict_if_present(self.cfg.trainer.model, model_and_diffusion_defaults().keys())
        )
        model_and_diffusion_args.update(
            args_to_dict_if_present(self.cfg.trainer.diffusion, model_and_diffusion_defaults().keys())
        )
        LOG.info("creating model and diffusion...")

        self.model, self.diffusion = create_model_and_diffusion(**model_and_diffusion_args)
        LOG.info("creating model and diffusion...")
        
        if self.cfg.trainer.gpu is not None:
            th.cuda.set_device(self.cfg.trainer.gpu)
            self.model.cuda(self.cfg.trainer.gpu)
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.cfg.trainer.gpu],
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            self.use_ddp = True
        else:
            LOG.error(f"No training on GPU possible on rank : {self.cfg.trainer.rank}, local_rank : {self.cfg.trainer.gpu}")
            raise NotImplementedError

        LOG.info("creating model and diffusion...")
        self.batch_size = self.cfg.trainer.batch_size
        self.microbatch = self.cfg.trainer.microbatch if self.cfg.trainer.microbatch > 0 else self.batch_size
        self.lr = self.cfg.trainer.lr
        self.ema_rate = (
            [self.cfg.trainer.ema_rate]
            if isinstance(self.cfg.trainer.ema_rate, float)
            else [float(x) for x in self.cfg.trainer.ema_rate.split(",")]
        )

        # TODO: fix this to be configurably in a proper way
        self.log_interval = self.cfg.trainer.log_interval
        self.save_interval = self.cfg.trainer.save_interval
        self.resume_checkpoint = self.cfg.trainer.checkpoint if 'checkpoint' in self.cfg.trainer else None
        self.use_fp16 = self.cfg.trainer.use_fp16
        self.fp16_scale_growth = 1e-3
        # self.schedule_sampler = self.cfg.trainer.schedule_sampler or UniformSampler(self.diffusion)
        self.schedule_sampler = None or UniformSampler(self.diffusion)
        self.weight_decay = self.cfg.trainer.weight_decay
        self.lr_anneal_steps = self.cfg.trainer.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        LOG.info(f"Loaded model checkpoint, i'm at resume step : {self.resume_step}.")
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        
        dist.barrier()
        LOG.info("Model initialized.")

        ######### Dataset creation
        if self.cfg.trainer.rank == 0:
            dataset = AudioDatasetExtended(self.cfg)
            if dataset.should_build_dataset():
                dataset.build_collated()
        dist.barrier()
        if self.cfg.trainer.rank != 0:
            dataset = AudioDatasetExtended(self.cfg)

        # self.data = load_data(
        #     data_dir = self.cfg.dataset.imgs.path,
        #     batch_size = self.cfg.trainer.batch_size,
        #     image_size = self.cfg.trainer.model.image_size,
        #     class_cond = self.cfg.trainer.class_cond,
        # )


        # TODO: add class_cond
        dataloader = create_dataloader(
            dataset,
            # ignoring the 'subset' and 'eq_mode' parameters for now
            subset="train",
            rank=self.cfg.trainer.rank,
            max_workers=4,
            world_size=self.cfg.trainer.world_size,
            batch_size=self.cfg.trainer.batch_size,
        )
        self.data = create_generator_from_dataloader(dataloader)

        LOG.info("Dataset initialized.")
        print("Dataset", len(dataset))
        LOG.info(f"Dataloader number of songs : {len(dataloader)}.")
        print(f"Train Dataloader number of songs : {len(dataloader)}.")
        
        dist.barrier()
        print("Trainer Initialization passed successfully.")
        LOG.info("Trainer Initialization passed successfully.")

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    th.load(resume_checkpoint, map_location=dist_util.dev())
                    # dist_util.load_state_dict(
                    #     resume_checkpoint, map_location=dist_util.dev()
                    # )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
                # state_dict = dist_util.load_state_dict(
                #     ema_checkpoint, map_location=dist_util.dev()
                # )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            # state_dict = dist_util.load_state_dict(
            #     opt_checkpoint, map_location=dist_util.dev()
            # )
            self.opt.load_state_dict(state_dict)

    def run(self):
        while (
            (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            )
            and self.step + self.resume_step < self.cfg.trainer.total_steps

        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
