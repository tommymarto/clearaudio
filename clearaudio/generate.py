import logging
import os

from pathlib import Path
from functools import lru_cache
from sys import modules

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
import torch.distributed as dist

from clearaudio.models.vqvae.vqvae_current import VQVAE
from clearaudio.models.wavenet.wavenet_encoder import WavenetEncoder
from clearaudio.models.wavenet.wavenet_decoder import WavenetDecoder
from clearaudio.models.wavenet.postnet import PostNet
from clearaudio.models.vqvae.vqvae_current import VQVAE
from clearaudio.transforms import signal
from clearaudio.datasets import audio
from clearaudio.models.guided_diffusion import logger
from clearaudio.models.guided_diffusion.script_util import args_to_dict_if_present, create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults
from clearaudio.models.guided_diffusion import dist_util
from clearaudio.trainers.base_trainer import init_local_distributed_mode

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
    modules = {}
    if cfg.trainer.mol.active:
        classes = cfg.trainer.mol.n_mols * 3
    else:
        classes = cfg.trainer.classes
    if cfg.trainer.type == "wavenet":
        try: 
            decoder = WavenetDecoder(cfg, classes=classes)
            decoder = DataParallel(decoder)
            decoder.load_state_dict(ckp['decoder_state_dict'])
            decoder.cuda()
            decoder.eval()
        except:
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
            LOG.warning('No encoder key found in the checkpoint.')

        try:
            postnet = PostNet(cfg)
            postnet = DataParallel(postnet)
            postnet.load_state_dict(ckp['postnet_state_dict'])
            postnet.cuda()
            postnet.eval()
        except:
            postnet = None
            LOG.warning('No postnet key found in the checkpoint.')
    
        modules['decoder'] = decoder
        modules['encoder'] = encoder
        modules['postnet'] = postnet
        modules['classes'] = classes
            
    elif cfg.trainer.type == 'vqvae':
        try:
            vqvae = VQVAE(cfg,output_classes=classes)
            vqvae.cuda()
            vqvae = DataParallel(vqvae)
            vqvae.load_state_dict(ckp['vqvae_state_dict'])
            vqvae.eval()
        except:
            vqvae = None
            LOG.warning("No vqvae key found in the checkpoint.")
        try:
            postnet = PostNet(cfg)
            postnet.cuda()
            postnet = DataParallel(postnet)
            postnet.load_state_dict(ckp['postnet_state_dict'])
            postnet.eval()
        except:
            postnet = None
            LOG.warning('No postnet key found in the checkpoint.')
        
        modules['vqvae'] = vqvae
        modules['postnet'] = postnet
    else:
        raise NotImplementedError

    return modules


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


def generate_song(lq_song_path: Path, output_dir: Path, cfg: DictConfig, modules: dict) -> None:
    waveform = load_audio(lq_song_path, cfg.dataset.sample_rate, cfg.generator.normalize_gain)
    if cfg.trainer.type == "wavenet":
        decoder, encoder, postnet, classes = modules['decoder'], modules['encoder'], modules['postnet'], modules['classes']
    elif cfg.trainer.type == 'vqvae':
        vqvae, postnet = modules['vqvae'], modules['postnet']
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
                if cfg.trainer.type == "wavenet":
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
    
    if cfg.generator.dump_recomposed_file:
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


def generate(input_path: Path, output_dir: Path, cfg: DictConfig, checkpoint_dict: dict) -> None:
    modules = load_modules(checkpoint_dict, cfg)

    if input_path.is_file():
        generate_song(input_path, output_dir, cfg, modules)
    elif input_path.is_dir():
        df = audio.find_audio_files(input_path, ['wav', 'mp3'])
        for _, p in df.iterrows():
            LOG.info(f"Restoring: {p['name']}")
            generate_song(Path(p['path']), output_dir, cfg, modules)
    else:
        LOG.error(f'Unknown input (neither file or dir??)')

def sample_diffusion(cfg: DictConfig) -> None:
    ckp = Path(cfg.generator.checkpointpath).expanduser()
    if not ckp.exists():
        LOG.error(f'Checkpoint not found: {ckp}')
        return 1

    LOG.info(f'Checkpoint found: {ckp}')
    LOG.info(f'Checkpoint dict keys')
    ckp = torch.load(ckp)
    
    # init_local_distributed_mode(cfg)
    # ckp = dist_util.load_state_dict(str(ckp), map_location="cpu")
    LOG.info(f'Checkpoint dict keys')

    logger.configure(cfg.generator.log_dir)


    logger.log("creating model and diffusion...")
    model_and_diffusion_args = model_and_diffusion_defaults()
    model_and_diffusion_args.update(
        args_to_dict_if_present(cfg.trainer.model, model_and_diffusion_defaults().keys())
    )
    model_and_diffusion_args.update(
        args_to_dict_if_present(cfg.trainer.diffusion, model_and_diffusion_defaults().keys())
    )
    model_and_diffusion_args.update(
        args_to_dict_if_present(cfg.generator.sampler, model_and_diffusion_defaults().keys())
    )

    LOG.info("Using config: %s", model_and_diffusion_args)

    model, diffusion = create_model_and_diffusion(**model_and_diffusion_args)
    
    model.load_state_dict(
        ckp
        # dist_util.load_state_dict(cfg.model_path, map_location="cpu")
    )

    LOG.info("loaded!")

    model.cuda()
    model.eval()

    LOG.info("Moved to cuda!")
    logger.log(f"{'reverse ' if cfg.generator.reverse_loop else ''}sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * cfg.trainer.batch_size < cfg.generator.num_samples:
        model_kwargs = {}
        # if cfg.trainer.class_cond:
        #     classes = th.randint(
        #         low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        #     )
        #     model_kwargs["y"] = classes

        if not cfg.generator.reverse_loop:
            sample_fn = (
                diffusion.p_sample_loop if not cfg.trainer.sampler.use_ddim else diffusion.ddim_sample_loop
            )

            noise = None
            if 'starting_noise' in cfg.generator:
                noise = torch.tensor(np.load(cfg.generator.starting_noise))

                if cfg.trainer.batch_size > 1:
                    noise = noise.repeat(cfg.trainer.batch_size, 1, 1, 1)
                LOG.info(f'Using starting noise from {cfg.generator.starting_noise}')

            sample = sample_fn(
                model,
                (cfg.trainer.batch_size, cfg.trainer.model.image_channels, cfg.trainer.model.image_size, cfg.trainer.model.image_size),
                clip_denoised=cfg.trainer.sampler.clip_denoised,
                noise=noise,
                model_kwargs=model_kwargs,
                progress=True,
            )

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
        else:
            sample_fn = diffusion.ddim_reverse_sample_loop

            image = np.load(cfg.generator.reverse_loop_image)
            x = np.array(image).astype(np.float32) / 127.5 - 1
            x = torch.tensor(np.transpose(x, (2, 0, 1)))
            x = x.unsqueeze(0)

            sample = sample_fn(
                model,
                x,
                model_kwargs=model_kwargs,
                progress=True,
            )




        all_images.extend([sample.cpu().numpy()])
        
        # NEED TORCH DIST
        # gathered_samples = [torch.zeros_like(sample) for _ in range(1)]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # if args.class_cond:
        #     gathered_labels = [
        #         th.zeros_like(classes) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(gathered_labels, classes)
        #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * cfg.trainer.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: cfg.generator.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{'reverse_' if cfg.generator.reverse_loop else ''}{shape_str}.npz")
    logger.log(f"saving to {out_path}")
        # if args.class_cond:
        #     np.savez(out_path, arr, label_arr)
        # else:
    np.savez(out_path, arr)

    # dist.barrier()
    logger.log("sampling complete")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    gen_cfg = cfg.generator
    input_path = Path(gen_cfg.input_path).expanduser()

    if cfg.trainer.type == 'diffusion':
        LOG.info('Unconditional generation from diffusion model')
        sample_diffusion(cfg)

        return 0

    if not input_path.exists():
        LOG.error(f'Invalid input path: {input_path}')
        return 1

    ckp = Path(gen_cfg.checkpointpath).expanduser()
    if not ckp.exists():
        LOG.error(f'Checkpoint not found: {ckp}')
        return 1

    ckp = torch.load(ckp)
    # override current config with internal config from the checkpoint
    if 'config' in ckp:
        cfg = OmegaConf.create(ckp['config'])
        with open_dict(cfg): # override the generator part of the conf stored in checkpoint
            cfg.generator = gen_cfg

    # LOG.info(f'Checkpoint dict keys: {ckp.keys()}')
    output_dir = Path(cfg.generator.output_dir).expanduser()
    return generate(input_path, output_dir, cfg, ckp)
    
if __name__ == "__main__":
    main()