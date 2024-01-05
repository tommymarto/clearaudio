from doctest import OutputChecker
from itertools import chain
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict, List, Tuple, Any, Optional
import copy
import os

import torch
import torch.cuda.amp as amp
from torch import Tensor, nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_value_
from torch.nn.parallel import DistributedDataParallel

import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import mu_law_encoding, mu_law_decoding


from torch.utils.tensorboard import SummaryWriter
import submitit

from clearaudio.models.wavenet.postnet import PostNet
from clearaudio.trainers.base_trainer import BaseTrainer
from clearaudio.datasets.audio import create_dataloader, AudioDataset
from clearaudio.models.vqvae.vqvae_current import VQVAE
from clearaudio.utils.wavenet_utils import (
    generate_waveform,
    snr, 
    si_sdr,
    make_melspectrogram_list,
    discretized_mix_logistic_loss,
    cross_entropy_loss,    
    spectral_loss, 
    compute_window_number,
    cross_entropy_loss,
    overlap_and_add_samples,
    cut_track_stack,
    plot_signal,
    linear_pcm,
)


LOG = logging.getLogger(__name__)

class VQVAE_Trainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def setup_trainer(self) -> None:
        LOG.info(f"VQVAE_Trainer: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}")
        torchaudio.set_audio_backend("sox_io")
        # if self.cfg.trainer.platform == "slurm":
        #     torchaudio.set_audio_backend('soundfile')

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

        if self.cfg.trainer.mol.active:
            self.classes = self.cfg.trainer.mol.n_mols * 3
            self.loss = discretized_mix_logistic_loss
        else:
            self.classes = self.cfg.trainer.classes
            if self.classes == 1:
                LOG.info('Using MSE loss')
                self.loss = nn.MSELoss()
                self.lambda_waveform = 500
            elif self.classes < 1:
                raise NotImplementedError
            else:
                LOG.info("Using CrossEntropy LossS")
                self.loss = cross_entropy_loss
        if self.cfg.trainer.melspec:
            self.mel_spectrogram_list = make_melspectrogram_list(multiple_stft = self.cfg.trainer.melspec.multiple_stft,stable = self.cfg.trainer.melspec.stable,log_scale = self.cfg.trainer.melspec.log_scale, sample_rate = self.sample_rate)

        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.trainer.gpu)
        else:
            LOG.error(f"No training on GPU possible on rank : {self.cfg.trainer.rank}, local_rank : {self.cfg.trainer.gpu}")
            raise NotImplementedError

        self.vqvae = VQVAE(self.cfg, output_classes = self.classes)
        self.vqvae.cuda(self.cfg.trainer.gpu)
        self.vqvae = DistributedDataParallel(self.vqvae, device_ids=[self.cfg.trainer.gpu])

        if self.cfg.trainer.postnet.active:
            self.postnet = PostNet(self.cfg)
            self.postnet.cuda(self.cfg.trainer.gpu)
            self.postnet = DistributedDataParallel(self.postnet, device_ids=[self.cfg.trainer.gpu])
            self.postnet_optimizer = optim.Adam(self.postnet.parameters(), lr = self.cfg.trainer.lr)
            self.postnet_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.postnet_optimizer, self.cfg.trainer.lr_decay)
            self.postnet_optimizer.zero_grad()
        else:
            self.postnet = None

        self.current_epoch = 0


        self.optimizer = optim.Adam(self.vqvae.parameters(),
                                        lr=self.cfg.trainer.lr)
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.trainer.lr_decay)

        LOG.info('Looking for existing checkpoints in results dir....')
        chk_path = self.find_latest_checkpoint_path(self.cfg.trainer.output_dir)
        if chk_path:
            LOG.info(f'Checkpoint found: {str(chk_path)}')
            print(f'Checkpoint found: {str(chk_path)}')
            checkpoint_dict = self.checkpoint_load(chk_path)
            self.vqvae.load_state_dict(checkpoint_dict["vqvae_state_dict"])
            if self.cfg.trainer.load_optimizer_dict.active:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
            self.current_epoch = checkpoint_dict["epoch"]
        else:
            LOG.info('No checkpoint found.')
        
        self.optimizer.zero_grad()
        if self.cfg.trainer.mixed_precision:
            self.scaler = amp.GradScaler(init_scale=self.cfg.trainer.init_scale)
        # Dataset Creation
        print("Dataset Creation")
        if self.cfg.trainer.rank == 0:
            dataset = AudioDataset(self.cfg)
            if dataset.should_build_dataset():
                dataset.build_collated()
        dist.barrier()
        if self.cfg.trainer.rank != 0:
            dataset = AudioDataset(self.cfg)
        print("Use random EQ : ", self.cfg.dataset.use_random_effects)
        self.dataset_train = copy.deepcopy(dataset)
        self.dataset_test = copy.deepcopy(dataset)
        self.dataset_test_on_train_music = copy.deepcopy(dataset)
        self.dataset_test_on_train_eq = copy.deepcopy(dataset)

        self.train_dataloader = create_dataloader(
            self.dataset_train,
            subset="train",
            eq_mode='random' if self.cfg.dataset.use_random_effects else "train",
            rank=self.cfg.trainer.rank,
            world_size=self.cfg.trainer.world_size,
            batch_size=self.cfg.trainer.batch_size,
        )
        self.test_dataloader = create_dataloader(
            self.dataset_test,
            subset="test",
            eq_mode="test",
            rank=self.cfg.trainer.rank,
            world_size=self.cfg.trainer.world_size,
            batch_size=self.cfg.trainer.batch_size,
        )

        self.test_on_train_music_dataloader = create_dataloader(
            self.dataset_test_on_train_music,
            subset="train",
            eq_mode="test",
            rank=self.cfg.trainer.rank,
            world_size=self.cfg.trainer.world_size,
            batch_size=self.cfg.trainer.batch_size,
        )      

        self.test_on_train_eq_dataloader = create_dataloader(
            self.dataset_test_on_train_eq,
            subset="test",
            eq_mode='random' if self.cfg.dataset.use_random_effects else "test",
            rank=self.cfg.trainer.rank,
            world_size=self.cfg.trainer.world_size,
            batch_size=self.cfg.trainer.batch_size,
        )  

        print("Dataset length :", len(dataset))
        LOG.info(f"Dataloader number of songs : {len(self.train_dataloader)}.")
        print(f"Train Dataloader number of songs : {len(self.train_dataloader)}.")
        print(f"Test Dataloader number of songs : {len(self.test_dataloader)}.")
        dist.barrier()
        print("Trainer Initialization passed successfully.")
        LOG.info("Trainer Initialization passed successfully.")

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        print("Requeuing SLURM job", OmegaConf.to_yaml(self.cfg))
        empty_trainer = type(self)(self.cfg)
        if self.vqvae is not None and \
            self.optimizer is not None and \
            self.scheduler is not None:
            
            self.checkpoint_dump(
                checkpoint_path=self.cfg.trainer.checkpointpath,
                epoch=self.current_epoch,
                vqvae_state_dict=self.vqvae.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
            )
        else:
            pass
        print("Sending Delayed Submission...")
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def checkpoint_dump(
        self,
        checkpoint_path: str = None,
        epoch: int = 0,
        vqvae_state_dict: Dict = None,
        transformer_state_dict: Dict = None,
        postnet_state_dict: Dict = None,
        optimizer_state_dict: Dict = None,
        scheduler_state_dict: Dict = None,
        **kwargs,
    ) -> None:
        if (vqvae_state_dict is None) and (self.vqvae is not None):
            vqvae_state_dict = self.vqvae.state_dict()
        if (postnet_state_dict is None) and (self.postnet is not None):
            postnet_state_dict = self.postnet.state_dict()
        if (optimizer_state_dict is None) and (self.optimizer is not None):
            optimizer_state_dict = self.optimizer.state_dict()
        if (scheduler_state_dict is None) and (self.scheduler is not None):
            scheduler_state_dict = self.scheduler.state_dict()

        ####    
        if checkpoint_path is None:
            prefix = self.cfg.trainer.output_dir
            if self.cfg.trainer.platform == "local": # Hydra changes base directory
                prefix = ''

            checkpoint_path = os.path.join(prefix, f"checkpoint_epoch_{str(epoch)}.pt")

        torch.save(
            {
                "epoch": epoch,
                "config": OmegaConf.to_container(self.cfg, resolve=False),
                "vqvae_state_dict": vqvae_state_dict,
                "transformer_state_dict": transformer_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "scheduler_state_dict": scheduler_state_dict,
                **kwargs,
            },
            checkpoint_path,
        )

    def track_SNR(self,inputs, targets, waveform_generated, waveform_post=None, iteration=0, n_batches=0, epoch=0, mode = 'train'):
        si_sdr_original = si_sdr(inputs[0], targets[0])
        si_sdr_generated = si_sdr(waveform_generated[0], targets[0])               
        snr_original = snr(inputs[0], targets[0])
        snr_generated = snr(waveform_generated[0], targets[0])
        if waveform_post is not None:
            si_sdr_post = si_sdr(waveform_post[0], targets[0])
            snr_post = snr(waveform_post[0], targets[0])
        if self.writer is not None:
            self.writer.add_scalar(f"{mode}_SDR/Inputs", si_sdr_original.item(), iteration + epoch*n_batches)
            self.writer.add_scalar(f"{mode}_SDR/Generated", si_sdr_generated.item(), iteration + epoch*n_batches)
            self.writer.add_scalar(f'{mode}_SNR/Inputs', snr_original.item(),iteration + epoch*n_batches)
            self.writer.add_scalar(f"{mode}_SNR/Generated", snr_generated.item(), iteration + epoch*n_batches)
            if waveform_post is not None:
                self.writer.add_scalar(f"{mode}_SDR/Post", si_sdr_post.item(), iteration + epoch*n_batches)
                self.writer.add_scalar(f'{mode}_SNR/Post', snr_post.item(), iteration + epoch*n_batches)

    def track_and_plot(self, epoch, iteration, inputs, targets,
                       waveform_generated,waveform_generated_post, loss, q_loss, commit_loss, recon_loss, 
                       melspec_loss,loss_postnet, eq_name = None, mode: str = 'train') -> None:
        if not self.writer:
            return
        if self.cfg.trainer.use_wandb:
            wandb.log({f"{mode}/Full_loss": loss})
        # only the master node passes this condition
        if eq_name is None:  # 👍
            eq_name = []
        if mode.lower() == "train": 
            real_iteration = epoch * self.cfg.trainer.epoch_len + iteration
        elif mode.lower() == "test" or mode.lower() == "test_on_train_eq" or mode.lower() == "test_on_train_music":
            real_iteration = epoch * self.cfg.trainer.eval_epoch_len + iteration
        else:
            LOG.error('Mode not recognized.')
            real_iteration = epoch * self.cfg.trainer.epoch_len + iteration
        self.writer.add_scalar(f'{mode}/Total_loss', loss.data.item(), real_iteration)
        self.writer.add_scalar(f'{mode}_Codebook/Q_loss', q_loss.data.item(), real_iteration)
        self.writer.add_scalar(f'{mode}_Codebook/Commit_loss', commit_loss.data.item(), real_iteration)
        if melspec_loss is not None:
            self.writer.add_scalar(f'{mode}/Melspec_loss', melspec_loss.data.item(), real_iteration)
            self.writer.add_scalar(f'{mode}/Waveform_loss', recon_loss.data.item(), real_iteration)  
        if loss_postnet is not None:
            self.writer.add_scalar(f'{mode}/PostNet_loss', loss_postnet.data.item(), real_iteration)

        if self.cfg.trainer.verbose and not (iteration % self.cfg.trainer.log_audio_frequency):
            self.writer.add_audio(
                f'{mode}/Inputs_{str(real_iteration)}_eq_{str(eq_name[0])}',
                inputs[0].squeeze().detach().cpu(), 
                sample_rate=self.sample_rate
            )
            self.writer.add_audio(
                f'{mode}/Targets_{str(real_iteration)}',
                targets[0].squeeze().detach().cpu(), 
                sample_rate=self.sample_rate
            )       
        #If Modified inputs/Targets
            if self.cfg.trainer.mu_law:
                self.writer.add_audio(
                    f'{mode}/inputs_mu_{str(real_iteration)}',
                    mu_law_decoding(inputs[0].squeeze().detach().cpu(), quantization_channels=256),
                    sample_rate=self.sample_rate
                )
                if not self.cfg.trainer.mol.active:
                    self.writer.add_audio(
                        f'{mode}/targets_mu_{str(real_iteration)}',
                        mu_law_decoding(targets[0].squeeze().detach().cpu(), quantization_channels=256),
                        sample_rate=self.sample_rate
                    )
                else: 
                    self.writer.add_audio(
                        f'{mode}/target_{str(real_iteration)}',
                        targets[0].squeeze().detach().cpu(),
                        sample_rate=self.sample_rate
                    ) 
            elif self.cfg.trainer.linear_pcm:
                self.writer.add_audio(
                    f'{mode}/lpcm_inputs_{str(real_iteration)}',
                    inputs[0].squeeze().detach().cpu(),
                    sample_rate=self.sample_rate
                )
                self.writer.add_audio(
                    f'{mode}/lpcm_targets_{str(real_iteration)}',
                    targets[0].squeeze().detach().cpu(),
                    sample_rate=self.sample_rate
                    )
            self.writer.add_audio(
                f'{mode}/generated_{str(real_iteration)}',
                waveform_generated[0].detach().cpu(),
                sample_rate=self.sample_rate
            )
            if waveform_generated_post is not None:
                self.writer.add_audio(f'{mode}/Post_generated{str(real_iteration)}',
                    waveform_generated_post[0].detach().cpu(),
                    sample_rate=self.sample_rate
                )
                if self.cfg.trainer.use_wandb:
                    wandb.log({f"{mode}_Post_generated_{str(real_iteration)}": 
                        wandb.Audio(waveform_generated_post[0].detach().cpu().numpy().reshape(-1), 
                            caption=f'{mode}_Post_generated{str(real_iteration)}', 
                            sample_rate=self.sample_rate)}
                    )
                    wandb.log({f'{mode}_generated_{str(real_iteration)}': wandb.Audio(waveform_generated[0].detach().cpu().numpy().reshape(-1), 
                        caption=f'{mode}_generated_{str(real_iteration)}', 
                        sample_rate=self.sample_rate)}
                    )
            if self.cfg.trainer.use_wandb:
                wandb.log({f'{mode}_inputs_{str(real_iteration)}_eq_{str(eq_name[0])}': 
                    wandb.Audio(inputs[0].squeeze().detach().cpu().numpy(), 
                    caption=f'{mode}_inputs_{str(real_iteration)}_eq_{str(eq_name[0])}', 
                    sample_rate=self.sample_rate)}
                )
                wandb.log({f'{mode}_targets_{str(real_iteration)}': 
                    wandb.Audio(targets[0].squeeze().detach().cpu().numpy(), 
                    caption=f'{mode}_targets_{str(real_iteration)}', 
                    sample_rate=self.sample_rate)}
                )
            
    def format_waveform(self, inputs, targets) -> Tuple[Tensor]:
        """Change bit depth and apply mu law encoding if needed"""
        if self.cfg.trainer.mu_law:
            inputs = mu_law_encoding(inputs, self.classes).float()
            if not self.cfg.trainer.mol.active:  
                targets = mu_law_encoding(targets, self.classes) 
        elif self.cfg.trainer.linear_pcm:
            inputs = linear_pcm(inputs, self.classes).float()
            targets = linear_pcm(targets, self.classes)
        elif self.cfg.trainer.classes > 1 and not self.cfg.trainer.mol.active:
            targets = mu_law_encoding(targets, self.classes) 
        else:
            pass
        return inputs, targets

    def next_looping(self, mode: str = "train"):
        if mode == "train":
            try:
                y = next(self.train_iterator)
                return y
            except StopIteration:
                self.train_iterator = iter(self.train_dataloader)
                y = next(self.train_iterator)
                return y
        elif mode== "test_on_train_music":
            try:
                y = next(self.test_on_train_music_iterator)
                return y
            except StopIteration:
                self.test_on_train_music_iterator = iter(self.test_on_train_music_dataloader)
                y = next(self.test_on_train_music_iterator)
                return y
        elif mode== "test_on_train_eq":
            try:
                y = next(self.test_on_train_eq_iterator)
                return y
            except StopIteration:
                self.test_on_train_eq_iterator = iter(self.test_on_train_eq_dataloader)
                y = next(self.test_on_train_eq_iterator)
                return y
        elif mode == "test":
            try:
                y = next(self.test_iterator)
                return y
            except StopIteration:
                self.test_iterator = iter(self.test_dataloader)
                y = next(self.test_iterator)
                return y
        elif mode == "full_sample":
            try:
                y = next(self.full_sample_iterator)
                return y
            except StopIteration:
                self.full_sample_iterator = iter(self.dataset_test)
                y = next(self.full_sample_iterator)
                return y
        else:
            raise NotImplementedError

    def train_on_batch(self, inputs, targets, eq_names_target,  iteration = 0, n_batches = 0, epoch = 0, mode='train'):
        
        if self.cfg.trainer.mixed_precision:
            with amp.autocast():
                outputs, zs, commit_loss, metric = self.vqvae(inputs)
        else:
            outputs, zs, commit_loss, metric = self.vqvae(inputs)
            
        if torch.isnan(outputs).sum():
            print(outputs)
        fit_loss = metric['fit']

        if self.cfg.trainer.verbose:
            if self.writer is not None:
                self.writer.add_histogram(f"{mode}/latent encoded", zs[0], iteration + n_batches * epoch)
        # Reconstruction Loss
        outputs = outputs.float()
        with amp.autocast(enabled=False):
            if self.cfg.trainer.mol.active: 
                if self.cfg.trainer.mol.num_classes_decay:
                    loss = self.loss(y_hat=outputs, y=targets, num_classes=int(self.cfg.trainer.mol.n_classes * self.cfg.trainer.mol.num_classes_decay **(epoch+1))).mean()
                else:
                    loss = self.loss(y_hat=outputs, y=targets, num_classes=self.cfg.trainer.mol.n_classes).mean()
                recon_loss = loss.clone()
            else:
                loss = self.loss(outputs.squeeze(1), targets).mean() * self.cfg.trainer.decoder.lambda_waveform
                recon_loss = loss.clone()
        if torch.isnan(loss).sum():
            print("loss", loss)
        # Generating Waveforms
        with amp.autocast(enabled=False):
            if self.cfg.trainer.melspec.active or self.cfg.trainer.postnet.active:
                waveform_generated = generate_waveform(self.cfg, outputs)
            else:
                with torch.no_grad():
                    waveform_generated = generate_waveform(self.cfg, outputs)

        # Melspec Loss
        if torch.isnan(waveform_generated).sum():
            print("waveform generated", waveform_generated, waveform_generated.dtype, waveform_generated.size())
        if self.cfg.trainer.melspec.active:  
            with amp.autocast(enabled=False):
                melspec_loss, melspec_loss_list = spectral_loss(waveform_generated.float(), targets.float(), mel_spectrogram_list = self.mel_spectrogram_list , 
                            scale = self.cfg.trainer.melspec.scale_melspec, loss= self.cfg.trainer.melspec.loss, use_db=self.cfg.trainer.melspec.use_db)
                LOG.info(f"{mode}/Melspec Loss : {melspec_loss}")
                if self.writer is not None:
                    try:
                        self.writer.add_histogram(f'{mode}/Melspec_loss_over_sftt', torch.tensor(melspec_loss_list), iteration + n_batches * epoch)
                        melspec_loss = melspec_loss * self.cfg.trainer.melspec.lambda_mel
                        loss += melspec_loss
                    except:
                        LOG.error(f"Error in Histogram, {melspec_loss_list}, {iteration + n_batches * epoch}")
                        print("Melspec loss list :", melspec_loss_list)
                        print(f"melspec_loss' {melspec_loss}")
                        melspec_loss = None
        else:
            melspec_loss = None

        # PostNet Loss
        if self.cfg.trainer.postnet.active:
            if (iteration + epoch * self.cfg.trainer.epoch_len > self.cfg.trainer.postnet.n_steps_before_activation) or ('test' in str(mode) and (epoch+1)* self.cfg.trainer.epoch_len > self.cfg.trainer.postnet.n_steps_before_activation):
                if self.cfg.trainer.mixed_precision:
                    with amp.autocast():
                        if self.cfg.trainer.postnet.push_back_gradients:
                            waveform_post = self.postnet(waveform_generated)
                        else:
                            waveform_post = self.postnet(waveform_generated.detach())
                else:
                    if self.cfg.trainer.postnet.push_back_gradients:
                        waveform_post = self.postnet(waveform_generated)
                    else:
                        waveform_post = self.postnet(waveform_generated.detach())
                if torch.isnan(waveform_post).sum():
                    print("waveform post", waveform_post, waveform_post.dtype, waveform_post.size())
                #### Spectral Loss
                with amp.autocast(enabled=False):
                    waveform_post = waveform_post.float()
                    loss_postnet, _ = spectral_loss(waveform_post, targets.float(), mel_spectrogram_list = self.mel_spectrogram_list , 
                            scale = self.cfg.trainer.melspec.scale_melspec, loss= self.cfg.trainer.melspec.loss, use_db=self.cfg.trainer.melspec.use_db)
                    loss_postnet = loss_postnet * self.cfg.trainer.melspec.lambda_mel
                    #### L1-Loss Waveform
                    loss_postnet_waveform = torch.abs(waveform_post - targets).mean()
                if self.writer is not None:
                    self.writer.add_scalar(f"{mode}/Postnet/Spectral_Loss", loss_postnet.item(), iteration + n_batches * epoch)
                    self.writer.add_scalar(f"{mode}/Postnet/Waveform_Loss", (loss_postnet_waveform * self.cfg.trainer.postnet.lambda_waveform).item(), iteration + n_batches * epoch)
                loss_postnet += loss_postnet_waveform * self.cfg.trainer.postnet.lambda_waveform
                loss_postnet = loss_postnet * self.cfg.trainer.postnet.lambda_post
                if self.writer is not None:
                    self.writer.add_scalar(f"{mode}/Postnet/Total_Loss", loss_postnet.item(), iteration + n_batches * epoch)
                if self.cfg.trainer.postnet.push_back_gradients:
                    loss += loss_postnet 
            else:
                loss_postnet = None
                waveform_post = None
        else:
            loss_postnet = None
            waveform_post = None
        loss += commit_loss
        return loss, fit_loss, commit_loss, recon_loss, melspec_loss, loss_postnet, waveform_generated, waveform_post

    def train(self) -> None:
        LOG.info(f'VQVAE trainer: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}')
        starting_epoch = self.current_epoch 
        if (self.cfg.trainer.rank == 0) and (self.cfg.trainer.use_wandb):
            if self.cfg.trainer.log_gradient_frequency:
                wandb.watch(self.vqvae, log_freq = self.cfg.trainer.log_gradient_frequency)
        for epoch in range(starting_epoch, self.cfg.trainer.num_epoch):
            n_batches = self.cfg.trainer.epoch_len
            self.vqvae.train()
            if self.cfg.trainer.postnet.active:
                self.postnet.train()
            self.current_epoch = epoch
            LOG.info(f"Starting Epoch : {epoch}")
            self.train_iterator = iter(self.train_dataloader)
            with tqdm(total=n_batches, desc='Train epoch %d' % epoch) as train_enum:
                for iteration in range(n_batches):
                    inputs, targets, eq_name = self.next_looping(mode='train')
                    inputs = inputs[:,:,:self.cfg.trainer.timesteps]
                    inputs = inputs.cuda(self.cfg.trainer.gpu)
                    targets = targets[:,:,:self.cfg.trainer.timesteps]
                    targets = targets.cuda(self.cfg.trainer.gpu)
                    if np.random.random() < self.cfg.trainer.autoencoding:
                        inputs = targets
                    inputs, targets = self.format_waveform(inputs, targets)
                    loss, q_loss, commit_loss, recon_loss, melspec_loss, loss_postnet, waveform_generated, waveform_post = self.train_on_batch(inputs, targets,eq_name, iteration, n_batches, epoch)
                    if self.cfg.trainer.postnet.active:
                        self.postnet_optimizer.zero_grad()
                    self.optimizer.zero_grad() 
                    if self.cfg.trainer.mixed_precision:
                        self.scaler.scale(loss).backward()
                        if self.cfg.trainer.unscale_gradient:
                            self.scaler.unscale_(self.optimizer)

                    else:
                        loss.backward()

                    if self.cfg.trainer.grad_clip > 0:
                        clip_grad_value_(self.vqvae.parameters(), self.cfg.trainer.grad_clip)
                        if self.cfg.trainer.postnet.active:
                            clip_grad_value_(self.postnet.parameters(), self.cfg.trainer.postnet.grad_clip)

                    if self.cfg.trainer.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if self.cfg.trainer.postnet.active and self.cfg.trainer.postnet.push_back_gradients:
                        self.postnet_optimizer.step()
                    elif self.cfg.trainer.postnet.active:
                        LOG.info('Optimizing PostNet independently.')
                        self.postnet_optimizer.zero_grad()
                        loss_postnet.backward()
                        clip_grad_value_(self.postnet.parameters(), self.cfg.trainer.postnet.grad_clip)
                        self.postnet_optimizer.step()
                    self.track_SNR(inputs, targets, waveform_generated, waveform_post, iteration, n_batches, epoch, mode = 'train')
                    self.track_and_plot(epoch, iteration, inputs, 
                        targets, waveform_generated,waveform_post, loss, q_loss, commit_loss, recon_loss, 
                        melspec_loss,loss_postnet, eq_name, mode='train'
                    )
                    train_enum.set_description(f'Train (loss: {loss.data.item():.4f}) epoch {epoch}')
                    train_enum.update()
            dist.barrier()
            self.eval()
            self.scheduler.step()
            if self.cfg.trainer.per_epoch:
                if self.cfg.trainer.rank == 0:
                    self.checkpoint_dump(epoch=epoch)
            dist.barrier()
            if self.cfg.trainer.rank == 0:
                self.checkpoint_dump(
                    checkpoint_path=None, # find the right path
                    epoch=self.current_epoch)
        print(f"{self.cfg.trainer.rank}:{self.cfg.trainer.gpu} training finished")

    def eval(self):
        dist.barrier()
        if self.cfg.trainer.postnet.active:
            self.postnet.eval()
        self.vqvae.eval()
        LOG.info("Evaluation starting...")
        n_batches_eval = self.cfg.trainer.eval_epoch_len
        self.test_iterator = iter(self.test_dataloader)
        self.test_on_train_music_iterator = iter(self.test_on_train_music_dataloader)
        self.test_on_train_eq_iterator = iter(self.test_on_train_eq_dataloader)
        for mode in ['test', 'test_on_train_music', 'test_on_train_eq']:
            dic_per_eq = {}
            sdr_per_eq = {}
            with tqdm(total=n_batches_eval, desc='Eval epoch %d' % self.current_epoch) as eval_enum, torch.no_grad():
                for iteration_eval in range(n_batches_eval):
                    inputs, targets, eq_names = self.next_looping(mode=mode)
                    inputs = inputs[:,:,:self.cfg.trainer.timesteps]
                    inputs = inputs.cuda(self.cfg.trainer.gpu)
                    targets = targets[:,:,:self.cfg.trainer.timesteps]
                    targets = targets.cuda(self.cfg.trainer.gpu)
                    if np.random.random() < self.cfg.trainer.autoencoding:
                        inputs = targets
                    inputs, targets = self.format_waveform(inputs, targets)
                    if self.cfg.trainer.mixed_precision:
                        with amp.autocast():
                            loss, q_loss, commit_loss, recon_loss, melspec_loss, loss_postnet, waveform_generated, waveform_post = self.train_on_batch(inputs, targets,eq_names, iteration_eval, n_batches_eval, self.current_epoch, mode=mode)
                    else:
                        loss, q_loss, commit_loss, recon_loss, melspec_loss, loss_postnet, waveform_generated, waveform_post = self.train_on_batch(inputs, targets,eq_names, iteration_eval, n_batches_eval, self.current_epoch, mode=mode)
                    self.track_SNR(inputs, targets, waveform_generated, waveform_post, iteration_eval, n_batches_eval, self.current_epoch, mode = mode)
                    self.track_and_plot(self.current_epoch, iteration_eval, inputs, 
                            targets, waveform_generated,waveform_post, loss, q_loss,commit_loss,recon_loss, 
                            melspec_loss,loss_postnet, eq_names, mode=mode
                        )
                    for h, eq_name in enumerate(eq_names):
                        if eq_name in dic_per_eq.keys():
                            if loss.dim() < 1 :
                                dic_per_eq[eq_name].append(loss.data.cpu().item())
                                sdr_per_eq[eq_name].append(si_sdr(waveform_generated[h], targets[h]).data.cpu().item())
                            else:
                                dic_per_eq[eq_name].append(loss[h].data.cpu().item())
                                sdr_per_eq[eq_name].append(si_sdr(waveform_generated[h], targets[h]).data.cpu().item())
                        else :
                            if loss.dim() < 1:
                                dic_per_eq[eq_name] = [loss.data.item()]
                                sdr_per_eq[eq_name] = [si_sdr(waveform_generated[h], targets[h]).data.cpu().item()]
                            else:
                                dic_per_eq[eq_name] = [loss[h].data.cpu().item()]
                                sdr_per_eq[eq_name] = [si_sdr(waveform_generated[h], targets[h]).data.cpu().item()]    
                    eval_enum.set_description(f'{mode} (loss: {loss:.3f}) epoch {self.current_epoch}')
                    eval_enum.update()
            if self.writer is not None:
                for eq_name in dic_per_eq.keys():
                    self.writer.add_scalar(f"Loss_EQ_{mode}/{eq_name}", np.mean(np.array(dic_per_eq[eq_name])),self.current_epoch) 
                    self.writer.add_scalar(f"SDR_EQ_{mode}/{eq_name}", np.mean(np.array(sdr_per_eq[eq_name])),self.current_epoch) 