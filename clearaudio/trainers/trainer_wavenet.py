from itertools import chain
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict, List, Tuple, Any, Optional
import copy

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

import PIL
import os

from torch.utils.tensorboard import SummaryWriter
import submitit

from clearaudio.trainers.base_trainer import BaseTrainer
from clearaudio.datasets.audio import create_dataloader, AudioDataset

from clearaudio.utils.wavenet_utils import (
    generate_waveform,
    snr, 
    si_sdr,
    make_melspectrogram_list,
    discretized_mix_logistic_loss,
    spectral_loss, 
    compute_window_number,
    cross_entropy_loss,
    cut_track_stack,
    plot_signal,
    linear_pcm,
)
from clearaudio.models.wavenet.wavenet_encoder import WavenetEncoder
from clearaudio.models.wavenet.wavenet_decoder import WavenetDecoder
from clearaudio.models.wavenet.wavenet_discriminator import ZDiscriminator
from clearaudio.models.wavenet.postnet import PostNet


LOG = logging.getLogger(__name__)

class WavenetTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def setup_trainer(self) -> None:
        LOG.info(f"WavenetTrainer: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}")
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
        if self.cfg.trainer.mol.active:
            self.classes = self.cfg.trainer.mol.n_mols * 3
            self.loss = discretized_mix_logistic_loss
        else:
            self.classes = self.cfg.trainer.classes
            if self.classes == 1:
                print('Using MSE loss')
                self.loss = nn.MSELoss()
                self.lambda_waveform = 500
            elif self.classes < 1:
                raise NotImplementedError
            else:
                print("Using CrossEntropy LossS")
                self.loss = cross_entropy_loss

        if self.cfg.trainer.melspec:
            self.mel_spectrogram_list = make_melspectrogram_list(multiple_stft = self.cfg.trainer.melspec.multiple_stft,stable = self.cfg.trainer.melspec.stable,log_scale = self.cfg.trainer.melspec.log_scale, sample_rate = self.sample_rate)
            self.melspec_loss = torch.nn.MSELoss()

        if self.cfg.trainer.gpu is not None:
            torch.cuda.set_device(self.cfg.trainer.gpu)
            if self.cfg.trainer.encoder.active:    
                self.encoder = WavenetEncoder(self.cfg)
                self.encoder.cuda(self.cfg.trainer.gpu)
            else:
                self.encoder = None

            if self.cfg.trainer.discriminator_latent.active:
                self.discriminator = ZDiscriminator(self.cfg)
                self.discriminator.cuda(self.cfg.trainer.gpu)
            else:
                self.discriminator = None

            if self.cfg.trainer.postnet.active:
                self.postnet = PostNet(self.cfg)
                self.postnet.cuda(self.cfg.trainer.gpu)
            else:
                self.postnet = None

            self.decoder = WavenetDecoder(self.cfg, classes = self.classes)
            self.decoder.cuda(self.cfg.trainer.gpu)
        else:
            LOG.error(f"No training on GPU possible on rank : {self.cfg.trainer.rank}, local_rank : {self.cfg.trainer.gpu}")
            raise NotImplementedError

        if self.cfg.trainer.encoder.active:
            self.encoder = DistributedDataParallel(self.encoder, device_ids=[self.cfg.trainer.gpu])

        if self.cfg.trainer.discriminator_latent.active:
            self.discriminator = DistributedDataParallel(self.discriminator, device_ids=[self.cfg.trainer.gpu])

        if self.cfg.trainer.postnet.active:
            self.postnet = DistributedDataParallel(self.postnet, device_ids=[self.cfg.trainer.gpu])
            self.postnet_optimizer = optim.Adam(self.postnet.parameters(), lr = self.cfg.trainer.lr)
            self.postnet_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.postnet_optimizer, self.cfg.trainer.lr_decay)
            self.postnet_optimizer.zero_grad()

        self.decoder = DistributedDataParallel(self.decoder, device_ids=[self.cfg.trainer.gpu])

        self.current_epoch = 0

        if self.cfg.trainer.encoder.active:
            self.optimizer = optim.Adam(chain(self.encoder.parameters(),
                                        self.decoder.parameters()),
                                        lr=self.cfg.trainer.lr)
        else:
            self.optimizer = optim.Adam(self.decoder.parameters(),
                                        lr=self.cfg.trainer.lr)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.trainer.lr_decay)

        if self.cfg.trainer.discriminator_latent.active:
            self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      lr=self.cfg.trainer.lr)
            self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, self.cfg.trainer.lr_decay)
            self.d_optimizer.zero_grad()

        if self.cfg.trainer.mixed_precision:
            self.scaler = amp.GradScaler()

        ######## Looking for checkpoints
        LOG.info('Looking for existing checkpoints in output dir...')
        chk_path = self.cfg.trainer.checkpointpath
        checkpoint_dict = self.checkpoint_load(chk_path)
        if checkpoint_dict:
            LOG.info(f'Checkpoint found: {str(chk_path)}')
            self.decoder.load_state_dict(checkpoint_dict["decoder_state_dict"])
            
            if self.cfg.trainer.encoder.active:
                self.encoder.load_state_dict(checkpoint_dict["encoder_state_dict"])
                self.optimizer = optim.Adam(
                    chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.cfg.trainer.lr
                )
                
            else:
                self.optimizer = optim.Adam(self.decoder.parameters(), lr=self.cfg.trainer.lr
                )
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.trainer.lr_decay)
            # Define and load parameters of the optimizer
            self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
            # Track the epoch at which the training stopped
            self.current_epoch = checkpoint_dict["epoch"]
        else:
            LOG.info('No checkpoint found.')
        self.optimizer.zero_grad()
        
        ######### Dataset creation
        if self.cfg.trainer.rank == 0:
            dataset = AudioDataset(self.cfg)
            if dataset.should_build_dataset():
                dataset.build_collated()
        dist.barrier()
        if self.cfg.trainer.rank != 0:
            dataset = AudioDataset(self.cfg)
        
        self.dataset_train = copy.deepcopy(dataset)
        self.dataset_test = copy.deepcopy(dataset)
        self.dataset_test_eq = copy.deepcopy(dataset)
        self.dataset_test_song = copy.deepcopy(dataset)
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

        self.test_on_train_eq_dataloader = create_dataloader(
            self.dataset_test_eq,
            subset="test",
            eq_mode="train",
            rank=self.cfg.trainer.rank,
            world_size=self.cfg.trainer.world_size,
            batch_size=self.cfg.trainer.batch_size,
        )

        self.test_on_train_songs_dataloader = create_dataloader(
            self.dataset_test_song,
            subset="train",
            eq_mode='random' if self.cfg.dataset.use_random_effects else "test",
            rank=self.cfg.trainer.rank,
            world_size=self.cfg.trainer.world_size,
            batch_size=self.cfg.trainer.batch_size,
        )

        print("Dataset", len(dataset))
        LOG.info(f"Dataloader number of songs : {len(self.train_dataloader)}.")
        print(f"Train Dataloader number of songs : {len(self.train_dataloader)}.")
        print(f"Test Dataloader number of songs : {len(self.test_dataloader)}.")
        print(f"Test Dataloader same EQs number of songs : {len(self.test_on_train_eq_dataloader)}.")
        print(f"Test Dataloader same songs number of songs : {len(self.test_on_train_songs_dataloader)}.")
        dist.barrier()
        print("Trainer Initialization passed successfully.")
        LOG.info("Trainer Initialization passed successfully.")

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        LOG.info("Requeuing SLURM job", OmegaConf.to_yaml(self.cfg))
        print("Requeuing SLURM job", OmegaConf.to_yaml(self.cfg))
        empty_trainer = type(self)(self.cfg)

        decoder_state_dict = self.decoder.state_dict() if self.decoder is not None else None
        encoder_state_dict = self.encoder.state_dict() if self.encoder is not None else None
        optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
        scheduler_state_dict = self.scheduler.state_dict() if self.scheduler is not None else None
        postnet_state_dict = self.postnet.state_dict() if self.postnet is not None else None

        self.checkpoint_dump(
            checkpoint_path=self.cfg.trainer.checkpointpath,
            epoch=self.current_epoch,
            encoder_state_dict=encoder_state_dict,
            decoder_state_dict=decoder_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            scheduler_state_dict=scheduler_state_dict,
            postnet_state_dict=postnet_state_dict
        )
        print("Sending Delayed Submission...")
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def checkpoint_dump(
        self,
        checkpoint_path: str = None,
        epoch: int = 0,
        encoder_state_dict: Dict = None,
        decoder_state_dict: Dict = None,
        optimizer_state_dict: Dict = None,
        scheduler_state_dict: Dict = None,
        postnet_state_dict: Dict = None,
        **kwargs,
    ) -> None:
        if (encoder_state_dict is None) and (self.encoder is not None):
            encoder_state_dict = self.encoder.state_dict()
        if (decoder_state_dict is None) and (self.decoder is not None):
            decoder_state_dict = self.decoder.state_dict()
        if (optimizer_state_dict is None) and (self.optimizer is not None):
            optimizer_state_dict = self.optimizer.state_dict()
        if (scheduler_state_dict is None) and (self.scheduler is not None):
            scheduler_state_dict = self.scheduler.state_dict()
        if (postnet_state_dict is None) and (self.postnet is not None):
            postnet_state_dict = self.postnet.state_dict()
        ####    
        if checkpoint_path is None:
            prefix = self.cfg.trainer.output_dir
            if self.cfg.trainer.platform == "local": # Hydra changes base directory
                prefix = ''

            checkpoint_path = os.path.join(prefix, f"checkpoint_epoch_{str(epoch)}.pt")

        torch.save(
            {
                "epoch": epoch,
                "encoder_state_dict": encoder_state_dict,
                "decoder_state_dict": decoder_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "scheduler_state_dict": scheduler_state_dict,
                "postnet_state_dict": postnet_state_dict,
                "config": OmegaConf.to_container(self.cfg, resolve=False)  # save config as well
            },
            checkpoint_path,
        )

    def track_and_plot(self, epoch, iteration, inputs, targets,
                       waveform_generated,waveform_generated_post, loss, recon_loss, 
                       melspec_loss,confusion_loss,loss_postnet, eq_name = None, mode: str = 'Train') -> None:
        if not self.writer:
            return
        if self.cfg.trainer.use_wandb:
            wandb.log({f"{mode}/Full_loss": loss})
        # only the master node passes this condition
        if eq_name is None:
            eq_name = []
        
        if mode.lower() == "train": 
            real_iteration = epoch * self.cfg.trainer.epoch_len + iteration
        elif mode.lower() == "test":
            real_iteration = epoch * self.cfg.trainer.eval_epoch_len + iteration
        else:
            LOG.debug('Mode not recognized.')
            real_iteration = epoch * self.cfg.trainer.epoch_len + iteration

        self.writer.add_scalar(f'{mode}/Total_loss', loss.data.item(), real_iteration)
        if melspec_loss is not None:
            self.writer.add_scalar(f'{mode}/Melspec_loss', melspec_loss.data.item(), real_iteration)
            self.writer.add_scalar(f'{mode}/Waveform_loss', recon_loss.data.item(), real_iteration)  
        if confusion_loss is not None:
            self.writer.add_scalar(f'{mode}/Confusion_loss', confusion_loss.data.item(), real_iteration)
        if loss_postnet is not None:
            self.writer.add_scalar(f'{mode}/PostNet_loss', loss_postnet.data.item(), real_iteration)

        if self.cfg.trainer.verbose and not (iteration % 1000):
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
            if self.cfg.trainer.use_wandb:            
                wandb.log({f'{mode}_generated_{str(real_iteration)}': wandb.Audio(waveform_generated[0].detach().cpu().numpy().reshape(-1), 
                        caption=f'{mode}_generated_{str(real_iteration)}', 
                        sample_rate=self.sample_rate)})
                wandb.log({f'{mode}_inputs_{str(real_iteration)}_eq_{str(eq_name[0])}': wandb.Audio(inputs[0].squeeze().detach().cpu().numpy(), 
                    caption=f'{mode}_inputs_{str(real_iteration)}_eq_{str(eq_name[0])}', 
                    sample_rate=self.sample_rate)})
                wandb.log({f'{mode}_targets_{str(real_iteration)}': wandb.Audio(targets[0].squeeze().detach().cpu().numpy(), 
                    caption=f'{mode}_targets_{str(real_iteration)}', 
                    sample_rate=self.sample_rate)})

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
        elif mode == "test_on_train_eq":
            try:
                y = next(self.test_on_train_eq_iterator)
                return y
            except StopIteration:
                self.test_on_train_eq_iterator = iter(self.test_on_train_eq_dataloader)
                y = next(self.test_on_train_eq_iterator)
                return y
        elif mode == "test_on_train_songs":
            try:
                y = next(self.test_on_train_songs_iterator)
                return y
            except StopIteration:
                self.test_on_train_songs_iterator = iter(self.test_on_train_songs_dataloader)
                y = next(self.test_on_train_songs_iterator)
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

    def train_on_batch(self, inputs, targets, eq_names_target,  iteration = 0, n_batches = 0, epoch = 0):
        if self.cfg.trainer.encoder.active:
            z = self.encoder(inputs)
            outputs = self.decoder(inputs, z)
            if iteration % 100 == 0 and self.cfg.trainer.verbose:
                if self.writer is not None:
                    self.writer.add_histogram("latent encoded", z[0], iteration + n_batches * epoch)
                    self.writer.add_histogram("latent encoded transposed", z.transpose(1,2)[0], iteration + n_batches * epoch)
        else:
            outputs = self.decoder(inputs)
        
        # Reconstruction Loss
        if self.cfg.trainer.mol.active: 
            if self.cfg.trainer.mol.num_classes_decay:
                loss = self.loss(y_hat=outputs, y=targets, num_classes=int(self.cfg.trainer.mol.n_classes * self.cfg.trainer.mol.num_classes_decay **(epoch+1))).mean()
            else:
                loss = self.loss(y_hat=outputs, y=targets, num_classes=self.cfg.trainer.mol.n_classes ).mean()
            recon_loss = loss.clone()
        else:
            loss = self.loss(outputs.squeeze(1), targets).mean() * self.cfg.trainer.decoder.lambda_waveform
            recon_loss = loss.clone()
        
        if self.cfg.trainer.melspec.active or self.cfg.trainer.postnet.active:
            waveform_generated = generate_waveform(self.cfg, outputs)
        else:
            with torch.no_grad():
                waveform_generated = generate_waveform(self.cfg, outputs)

        # Melspec Loss
        if self.cfg.trainer.melspec.active: 
            with amp.autocast(enabled=False):
                melspec_loss, melspec_loss_list = spectral_loss(waveform_generated.float(), targets.float(), mel_spectrogram_list = self.mel_spectrogram_list , 
                            scale = self.cfg.trainer.melspec.scale_melspec, loss= self.cfg.trainer.melspec.loss, use_db=self.cfg.trainer.melspec.use_db)
            LOG.info(f"Melspec Loss : {melspec_loss}")
            if self.writer is not None:
                self.writer.add_histogram('train/Melspec_loss_over_sftt', torch.tensor(melspec_loss_list), iteration + n_batches * epoch)
            melspec_loss = melspec_loss * self.cfg.trainer.melspec.lambda_mel
            loss += melspec_loss
        else:
            melspec_loss = None
        
        # PostNet Loss
        if self.cfg.trainer.postnet.active:
            if iteration + epoch * self.cfg.trainer.epoch_len > self.cfg.trainer.postnet.n_steps_before_activation:
                if self.cfg.trainer.postnet.push_back_gradients:
                    waveform_post = self.postnet(waveform_generated)
                else:
                    waveform_post = self.postnet(waveform_generated.detach())

                #### Spectral Loss
                with amp.autocast(enabled=False):
                    loss_postnet, _ = spectral_loss(waveform_post.float(), targets.float(), mel_spectrogram_list = self.mel_spectrogram_list , 
                            scale = self.cfg.trainer.melspec.scale_melspec, loss= self.cfg.trainer.melspec.loss, use_db=self.cfg.trainer.melspec.use_db)
                loss_postnet = loss_postnet * self.cfg.trainer.melspec.lambda_mel
                #### L1-Loss Waveform
                loss_postnet_waveform = torch.abs(waveform_post - targets).mean()
                if self.writer is not None:
                    self.writer.add_scalar("Postnet/Spectral_Loss", loss_postnet.item(), iteration + n_batches * epoch)
                    self.writer.add_scalar("Postnet/Waveform_Loss", (loss_postnet_waveform * self.cfg.trainer.postnet.lambda_waveform).item(), iteration + n_batches * epoch)
                loss_postnet += loss_postnet_waveform * self.cfg.trainer.postnet.lambda_waveform
                loss_postnet = loss_postnet * self.cfg.trainer.postnet.lambda_post
                if self.writer is not None:
                    LOG.info(f"Postnet - Spectral_Loss {loss_postnet.item()}")
                    LOG.info(f"Postnet - Waveform_Loss {loss_postnet_waveform * self.cfg.trainer.postnet.lambda_waveform}")
                    self.writer.add_scalar("Postnet/Total_Loss", loss_postnet.item(), iteration + n_batches * epoch)
                if self.cfg.trainer.postnet.push_back_gradients:
                    loss += loss_postnet 
            else:
                loss_postnet = None
                waveform_post = None
        else:
            loss_postnet = None
            waveform_post = None

        #Confusion Loss
        if self.cfg.trainer.discriminator_latent.active:
            z_logits = self.discriminator(z)
            confusion_loss = -F.cross_entropy(z_logits, eq_names_target).mean()
            loss += confusion_loss * self.cfg.trainer.discriminator_latent.lambda_d
        else:
            confusion_loss = None
        print(f"{self.cfg.trainer.rank}:{self.cfg.trainer.gpu} - loss: {loss}")
        return loss, recon_loss,  melspec_loss, loss_postnet, confusion_loss , waveform_generated, waveform_post

    def train(self) -> None:
        print(f'Wavenet trainer: {self.cfg.trainer.rank}, gpu: {self.cfg.trainer.gpu}')
        n_batches = self.cfg.trainer.epoch_len
        starting_epoch = self.current_epoch
        if self.cfg.trainer.rank == 0:
            if self.cfg.trainer.use_wandb and self.cfg.trainer.log_gradient_frequency:
                wandb.watch(self.decoder, log_freq=self.cfg.trainer.log_gradient_frequency)
        for epoch in range(starting_epoch, self.cfg.trainer.num_epoch):
            if self.cfg.trainer.encoder.active:
                self.encoder.train()
            if self.cfg.trainer.postnet.active:
                self.postnet.train()
            self.decoder.train()
            self.current_epoch = epoch
            print(f"Starting Epoch : {epoch}")
            LOG.info(f"Starting Epoch : {epoch}")
            self.train_iterator = iter(self.train_dataloader)
            with tqdm(total=n_batches, desc='Train epoch %d' % epoch) as train_enum:
                for iteration in range(n_batches):
                    inputs, targets, eq_names = self.next_looping(mode='train')
                    inputs = inputs.cuda(self.cfg.trainer.gpu)
                    targets = targets.cuda(self.cfg.trainer.gpu)
                    if np.random.random() < self.cfg.trainer.autoencoding:
                        inputs = targets
                    inputs, targets = self.format_waveform(inputs, targets)

                    if self.cfg.trainer.discriminator_latent.active:
                        eq_names_target = torch.tensor(self.dataset_train.get_eq_indices(eq_names)).long().cuda(self.cfg.trainer.gpu)
                        z = self.encoder(inputs)
                        z_logits = self.discriminator(z)
                        discriminator_right = F.cross_entropy(z_logits, eq_names_target).mean()
                        loss = discriminator_right * self.cfg.trainer.discriminator_latent.lambda_d
                        self.d_optimizer.zero_grad()
                        loss.backward()
                        if self.cfg.trainer.grad_clip is not None:
                            clip_grad_value_(self.discriminator.parameters(), self.cfg.trainer.grad_clip)

                        self.d_optimizer.step()
                        

                    if self.cfg.trainer.mixed_precision:
                        with amp.autocast():
                            loss, recon_loss, melspec_loss, loss_postnet,confusion_loss,waveform_generated,waveform_post = self.train_on_batch(inputs, targets,eq_names, iteration, n_batches, epoch)
                    else:
                        loss,recon_loss,melspec_loss,loss_postnet,confusion_loss,waveform_generated,waveform_post = self.train_on_batch(inputs, targets,eq_names, iteration, n_batches, epoch)
                    
                    if self.cfg.trainer.postnet.active:
                        self.postnet_optimizer.zero_grad()

                    if self.cfg.trainer.mixed_precision:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.optimizer.zero_grad()

                    else:
                        self.optimizer.zero_grad() 
                        loss.backward()

                        if self.cfg.trainer.grad_clip > 0:
                            if self.cfg.trainer.encoder.active:
                                clip_grad_value_(self.encoder.parameters(), self.cfg.trainer.grad_clip)
                            clip_grad_value_(self.decoder.parameters(), self.cfg.trainer.grad_clip)
                            if self.cfg.trainer.postnet.active:
                                clip_grad_value_(self.postnet.parameters(), self.cfg.trainer.postnet.grad_clip)

                        if self.cfg.trainer.test_grad:
                            encoder_params = [x for x in self.encoder.parameters()]
                            decoder_params = [x for x in self.decoder.parameters()]

                        self.optimizer.step()

                        if self.cfg.trainer.test_grad:
                            LOG.info(
                                "Encoder updated : {}".format(encoder_params == [x for x in self.encoder.parameters()])
                            )
                            LOG.info(
                                "Decoder updated : {}".format(decoder_params == [x for x in self.decoder.parameters()])
                            )
                            if self.cfg.trainer.melspec.active:
                                LOG.info(
                                    "Waveform Generated require gradient : {}".format(waveform_generated.requires_grad)
                                    )

                    if (not self.cfg.trainer.postnet.push_back_gradients) and (loss_postnet is not None):
                        if self.cfg.trainer.mixed_precision:
                            self.scaler.scale(loss_postnet).backward()
                            self.scaler.step(self.postnet_optimizer)
                            self.postnet_optimizer.zero_grad()
                        else:
                            LOG.info('Optimizing PostNet independently.')
                            self.postnet_optimizer.zero_grad()
                            loss_postnet.backward()
                            clip_grad_value_(self.postnet.parameters(), self.cfg.trainer.postnet.grad_clip)
                            self.postnet_optimizer.step()
                    elif loss_postnet is not None :
                        if self.cfg.trainer.mixed_precision:
                            self.scaler.step(self.postnet_optimizer)
                            self.postnet_optimizer.zero_grad()
                        else:
                            self.postnet_optimizer.step()
                            
                            
                    if self.cfg.trainer.mixed_precision:
                        self.scaler.update()
                    si_sdr_original = si_sdr(inputs[0], targets[0])
                    si_sdr_generated = si_sdr(waveform_generated[0], targets[0])               
                    snr_original = snr(inputs[0], targets[0])
                    snr_generated = snr(waveform_generated[0], targets[0])

                    if loss_postnet is not None:
                        si_sdr_post = si_sdr(waveform_post[0], targets[0])
                        snr_post = snr(waveform_post[0], targets[0])
                    if self.writer is not None:
                        self.writer.add_scalar("train_SDR/Inputs", si_sdr_original.item(), iteration + epoch*n_batches)
                        self.writer.add_scalar("train_SDR/Generated", si_sdr_generated.item(), iteration + epoch*n_batches)
                        self.writer.add_scalar('train_SNR/Inputs', snr_original.item(),iteration + epoch*n_batches)
                        self.writer.add_scalar("train_SNR/Generated", snr_generated.item(), iteration + epoch*n_batches)
                        if loss_postnet is not None:
                            self.writer.add_scalar("train_SDR/Post", si_sdr_post.item(), iteration + epoch*n_batches)
                            self.writer.add_scalar('train_SNR/Post', snr_post.item(), iteration + epoch*n_batches)
                            
                    self.track_and_plot(epoch, iteration, inputs, 
                        targets, waveform_generated,waveform_post, loss, recon_loss, 
                        melspec_loss, confusion_loss,loss_postnet, eq_names, mode='train'
                    )
                    train_enum.set_description(f'Train (loss: {loss.data.item():.4f}) epoch {epoch}')
                    train_enum.update()
        
            self.eval()
            self.scheduler.step()
            if self.cfg.trainer.discriminator_latent.active:
                self.d_scheduler.step()
            if self.cfg.trainer.per_epoch:
                if self.cfg.trainer.rank == 0:
                    self.checkpoint_dump(
                        epoch=self.current_epoch
                    )
        dist.barrier()
        if self.cfg.trainer.rank == 0:
            self.checkpoint_dump(
                checkpoint_path=self.cfg.trainer.checkpointpath, # find the right path
                epoch=self.current_epoch,
                encoder_state_dict=self.encoder.state_dict(),
                decoder_state_dict=self.decoder.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
            )
        print(f"{self.cfg.trainer.rank}:{self.cfg.trainer.gpu} training finished")

    def eval(self):
        dist.barrier()
        if self.cfg.trainer.encoder.active:
            self.encoder.eval()
        self.decoder.eval()
        if self.cfg.trainer.postnet.active:
            self.postnet.eval()
        LOG.info("Evaluation starting...")
        n_batches = self.cfg.trainer.eval_epoch_len
        self.test_iterator = iter(self.test_dataloader)
        self.test_on_train_eq_iterator = iter(self.test_on_train_eq_dataloader)
        self.test_on_train_songs_iterator = iter(self.test_on_train_songs_dataloader)
        for mode in ['test', 'test_on_train_eq', "test_on_train_songs"]:
            dic_per_eq = {}
            sdr_per_eq = {}
            LOG.info(f"Starting evaluation in mode : {mode}")
            print(f"Starting evaluation in mode : {mode}")
            with tqdm(total=n_batches, desc='Eval epoch %d' % self.current_epoch) as eval_enum, torch.no_grad():
                for iteration in range(n_batches):
                    inputs, targets, eq_names = self.next_looping(mode=mode)
                    inputs = inputs.cuda(self.cfg.trainer.gpu)
                    targets = targets.cuda(self.cfg.trainer.gpu)
                    if np.random.random() < self.cfg.trainer.autoencoding:
                        inputs = targets
                    inputs, targets = self.format_waveform(inputs, targets)
                    if self.cfg.trainer.encoder.active:
                        z = self.encoder(inputs)
                        outputs = self.decoder(inputs, z)
                    else:
                        outputs = self.decoder(inputs)
                    # Reconstruction Loss
                    if self.cfg.trainer.mol.active:
                        loss = self.loss(y_hat=outputs, y=targets, num_classes=self.cfg.trainer.mol.n_classes).mean()
                        recon_loss = loss.clone()
                    else:
                        loss = self.loss(outputs.squeeze(1), targets).mean() * self.cfg.trainer.decoder.lambda_waveform
                        recon_loss = loss.clone()

                    if self.cfg.trainer.melspec.active or self.cfg.trainer.postnet.active:
                            waveform_generated = generate_waveform(self.cfg, outputs)
                    else:
                        with torch.no_grad():
                            waveform_generated = generate_waveform(self.cfg, outputs)

                    if self.cfg.trainer.melspec.active:
                        waveform_generated = generate_waveform(self.cfg, outputs)
                        melspec_loss, _ = spectral_loss(waveform_generated, targets, mel_spectrogram_list = self.mel_spectrogram_list , 
                                        scale = self.cfg.trainer.melspec.scale_melspec, loss= self.cfg.trainer.melspec.loss, use_db=self.cfg.trainer.melspec.use_db)
                        melspec_loss = melspec_loss * self.cfg.trainer.melspec.lambda_mel 
                        loss += melspec_loss           
                    else:
                        melspec_loss = None

                    # PostNet Loss
                    if self.cfg.trainer.postnet.active:
                        if (self.current_epoch+1) * self.cfg.trainer.epoch_len > self.cfg.trainer.postnet.n_steps_before_activation:
                            if self.cfg.trainer.postnet.push_back_gradients:
                                waveform_post = self.postnet(waveform_generated)
                            else:
                                waveform_post = self.postnet(waveform_generated.detach())
                            #### Spectral Loss
                            loss_postnet, _ = spectral_loss(waveform_post, targets, mel_spectrogram_list = self.mel_spectrogram_list , 
                                        scale = self.cfg.trainer.melspec.scale_melspec, loss= self.cfg.trainer.melspec.loss, use_db=self.cfg.trainer.melspec.use_db)
                            loss_postnet = loss_postnet * self.cfg.trainer.melspec.lambda_mel
                            #### L1-Loss Waveform
                            if self.cfg.trainer.postnet.tanh_loss_active:
                                loss_postnet_waveform = torch.tanh(torch.abs(waveform_post - targets)).mean()
                            else:
                                loss_postnet_waveform = torch.abs(waveform_post - targets).mean()
                            loss_postnet += loss_postnet_waveform 
                            loss += loss_postnet

                        else:
                            loss_postnet = None
                            waveform_post = None
                    else:
                        loss_postnet = None
                        waveform_post = None

                    confusion_loss = None


                    si_sdr_original = si_sdr(inputs[0], targets[0])
                    si_sdr_generated = si_sdr(waveform_generated[0], targets[0])
                    snr_original = snr(inputs[0], targets[0])
                    snr_generated = snr(waveform_generated[0], targets[0])
                    if loss_postnet is not None:
                        si_sdr_post = si_sdr(waveform_post[0], targets[0])
                        snr_post = snr(waveform_post[0], targets[0])
                    if self.writer is not None:
                        self.writer.add_scalar(f"{mode}_SDR/Inputs", si_sdr_original.item(), iteration + self.current_epoch *n_batches)
                        self.writer.add_scalar(f"{mode}_SDR/Generated", si_sdr_generated.item(), iteration + self.current_epoch*n_batches)
                        self.writer.add_scalar(f'{mode}_SNR/Inputs', snr_original.item(),iteration + self.current_epoch*n_batches)
                        self.writer.add_scalar(f"{mode}_SNR/Generated", snr_generated.item(), iteration + self.current_epoch*n_batches)
                        if loss_postnet is not None:
                            self.writer.add_scalar(f"{mode}_SDR/Post", si_sdr_post.item(), iteration + self.current_epoch*n_batches)
                            self.writer.add_scalar(f"{mode}_SNR/Post", snr_post.item(), iteration + self.current_epoch*n_batches)

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
                    self.track_and_plot(self.current_epoch, iteration, inputs, targets, waveform_generated,waveform_post, loss, recon_loss, melspec_loss, confusion_loss,loss_postnet, eq_names, mode=mode)
                    eval_enum.set_description(f'{mode} (loss: {loss:.3f}) epoch {self.current_epoch}')
                    eval_enum.update()
            if self.writer is not None:
                for eq_name in dic_per_eq.keys():
                    self.writer.add_scalar(f"Loss_EQ_{mode}/{eq_name}", np.mean(np.array(dic_per_eq[eq_name])),self.current_epoch) 
                    self.writer.add_scalar(f"SDR_EQ_{mode}/{eq_name}", np.mean(np.array(sdr_per_eq[eq_name])),self.current_epoch) 
        dist.barrier()
        

        


            



                  
