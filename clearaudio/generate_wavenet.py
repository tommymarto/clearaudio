import logging

from pathlib import Path
from clearaudio.models.vqvae.vqvae_current import VQVAE

import hydra
import submitit

from tqdm import tqdm
from omegaconf import DictConfig, open_dict, OmegaConf 
import numpy as np

import torch 
import torch.nn as nn
import torchaudio
from torchaudio.functional import mu_law_encoding, mu_law_decoding
from torch.distributions.categorical import Categorical

from torch.nn.parallel import DistributedDataParallel, DataParallel

from clearaudio.models.wavenet.wavenet_encoder import WavenetEncoder
from clearaudio.models.wavenet.wavenet_decoder import WavenetDecoder
from clearaudio.models.wavenet.postnet import PostNet
from clearaudio.models.vqvae.vqvae_current import VQVAE
from clearaudio.transforms import signal

from clearaudio.utils.wavenet_utils import (
    overlap_and_add_samples,
    cut_track_stack,
    linear_pcm,
    generate_waveform
)

LOG = logging.getLogger(__name__)


def format_waveform(inputs: torch.Tensor, classes: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Change bit depth and apply mu law encoding if needed"""
    if cfg.trainer.mu_law:
        inputs = mu_law_encoding(inputs, classes).float()
    elif cfg.trainer.linear_pcm:
        inputs = linear_pcm(inputs, classes).float()
    else:
        pass
    return inputs


def load_modules(ckp: dict, cfg: DictConfig):
    # READ from internal CFG with model
    if cfg.trainer.mol.active:
        classes = cfg.trainer.mol.n_mols * 3
    else:
        classes = cfg.trainer.classes
    if cfg.trainer.type == "wavenet" :
        try: 
            decoder = WavenetDecoder(cfg, classes=classes)
            decoder = DataParallel(decoder)
            decoder.load_state_dict(ckp['decoder_state_dict'])
            decoder.cuda()
            decoder.eval()
        except :
            LOG.error('No decoder key found in the checkpoint.')
            raise RuntimeError

        try:
            encoder = WavenetEncoder(cfg)
            encoder = DataParallel(encoder)
            encoder.load_state_dict(ckp['encoder_state_dict'])
            encoder.cuda()
            encoder.eval()
        except:
            encoder = None
            LOG.info('No encoder key found in the checkpoint.')

        try:
            postnet = PostNet(cfg)
            postnet = DataParallel(postnet)
            postnet.load_state_dict(ckp['postnet_state_dict'])
            postnet.cuda()
            postnet.eval()
        except:
            postnet = None
            LOG.info('No postnet key found in the checkpoint.')
    
        return decoder, encoder, postnet, classes
            
    elif cfg.trainer.type == 'vqvae':
        try:
            vqvae = VQVAE(cfg,output_classes=classes)
            vqvae.cuda()
            vqvae = DataParallel(vqvae)
            vqvae.load_state_dict(ckp['vqvae_state_dict'])
            vqvae.eval()
        except:
            vqvae = None
            LOG.info("No vqvae key found in the checkpoint.")
        try:
            postnet = PostNet(cfg)
            postnet.cuda()
            postnet = DataParallel(postnet)
            postnet.load_state_dict(ckp['postnet_state_dict'])
            postnet.eval()
        except:
            postnet = None
            LOG.info('No postnet key found in the checkpoint.')
        return vqvae, postnet
    else:
        raise NotImplementedError


def load_audio(path: Path, sample_rate: int, normalize_gain: bool):
    waveform, sr = torchaudio.load(path, normalize=True)
    if sr != sample_rate:
        LOG.warning('Input song has a different sampling rate than the model, resampling...')
        waveform, _ = signal.resample_signal(waveform, sr, sample_rate)
    waveform = signal.signal_to_mono(waveform)

    if normalize_gain:
        sox = signal.SoxEffectTransform()
        sox.normalize_gain()   
        waveform, _ = sox.apply_tensor(waveform, sample_rate)
    return waveform


def generate(lq_song_path: Path, output_dir: Path, cfg: DictConfig, checkpoint_dict: dict):
    waveform = load_audio(lq_song_path, cfg.dataset.sample_rate, cfg.generator.normalize_gain)
    if cfg.trainer.type == "wavenet" :
        decoder, encoder, postnet, classes = load_modules(checkpoint_dict, cfg)
    elif cfg.trainer.type == 'vqvae':
        vqvae, postnet = load_modules(checkpoint_dict, cfg)
    else:
        raise NotImplementedError

    window_length = cfg.dataset.sample_rate * cfg.generator.window_clip_duration
    overlap = 0.5
    inputs = waveform
    inputs_stacked = cut_track_stack(inputs, window_length=window_length, overlap=overlap).cpu()
    outputs_stacked = torch.zeros_like(inputs_stacked).cpu()
    outputs_stacked_post = torch.zeros_like(inputs_stacked).cpu()
    with torch.no_grad():
        for batch_idx in range(len(inputs_stacked)):
            for i, input in tqdm(enumerate(inputs_stacked[batch_idx])):
                input = input.cuda()
                input = format_waveform(input, classes, cfg)
                input = input.unsqueeze(0)
                if cfg.trainer.type == "wavenet" :
                    if encoder is not None:
                        z = encoder(input)
                        outputs = decoder(input, z)
                    else:
                        outputs = decoder(input)

                elif cfg.trainer.type == 'vqvae':
                    outputs, zs, commit_loss, metric = vqvae(input)

                else: 
                    raise NotImplementedError
                waveform_generated = generate_waveform(cfg, outputs)
                if postnet is not None:
                    waveform_post = postnet(waveform_generated)
                else:
                    waveform_post = None
                
                outputs_stacked[batch_idx, i] = waveform_generated.squeeze().cpu()
                outputs_stacked_post[batch_idx, i] = waveform_post.squeeze().cpu()

        equalized_full = overlap_and_add_samples(inputs_stacked,
                                                 overlap=overlap, 
                                                 window_length=window_length)
        generated_full = overlap_and_add_samples(outputs_stacked,
                                                 overlap=overlap, 
                                                 window_length=window_length)
        generated_post = overlap_and_add_samples(outputs_stacked_post,
                                                 overlap=overlap, 
                                                 window_length=window_length)

        song_name = lq_song_path.stem
        output_dir.mkdir(exist_ok=True)
        eq_song_path = (output_dir / f"{song_name}-comp.wav").expanduser().resolve()
        generated_song_path = (output_dir / f"{song_name}-generated.wav").expanduser().resolve()
        generated_post_song_path = (output_dir / f"{song_name}-post.wav").expanduser().resolve()
    
    torchaudio.save(eq_song_path, 
                    equalized_full, 
                    cfg.dataset.sample_rate)

    torchaudio.save(generated_song_path, 
                    generated_full, 
                    cfg.dataset.sample_rate)

    if postnet is not None:
        torchaudio.save(generated_post_song_path, 
                        generated_post, 
                        cfg.dataset.sample_rate)



@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    gen_cfg = cfg.generator
    lq_song_path = Path(gen_cfg.song_path).expanduser()

    if not lq_song_path.exists():
        LOG.error('Low quality song not found...')
        return 1

    ckp = Path(gen_cfg.checkpointpath).expanduser()
    if not ckp.exists():
        LOG.error('Checkpoint not found.')
        return 1

    ckp = torch.load(ckp)
    # override current config with internal config from the checkpoint
    if 'config' in ckp:
        cfg = OmegaConf.create(ckp['config'])
        with open_dict(cfg): # override the generator part of the conf stored in checkpoint
            cfg.generator = gen_cfg

    LOG.info(f'Checkpoint dict keys: {ckp.keys()}')
    return generate(lq_song_path, cfg.generator.output_dir, cfg, ckp)
    
if __name__ == "__main__":
    main()