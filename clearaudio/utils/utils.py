import pathlib
import urllib
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torchaudio.transforms as T
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor
import gc

def get_output_dir(cfg, job_id=None) -> Path:
    out_dir = Path(cfg.trainer.output_dir)
    p = Path(out_dir).expanduser()
    if job_id is not None:
        p = p / str(job_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_key_value_to_conf(cfg: DictConfig, key: Any, value: Any) -> DictConfig:
    with open_dict(cfg):
        cfg[key] = value
    return cfg


def fix_random_seeds(seed: int = 31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def file_uri_to_path(file_uri: str, path_class=Path) -> Path:
    # https://stackoverflow.com/questions/5977576/is-there-a-convenient-way-to-map-a-file-uri-to-os-path
    """
    This function returns a pathlib.PurePath object for the supplied file URI.

    :param str file_uri: The file URI ...
    :param class path_class: The type of path in the file_uri. By default it uses
        the system specific path pathlib.PurePath, to force a specific type of path
        pass pathlib.PureWindowsPath or pathlib.PurePosixPath
    :returns: the pathlib.PurePath object
    :rtype: pathlib.PurePath
    """
    windows_path = isinstance(path_class(), pathlib.PureWindowsPath)
    file_uri_parsed = urllib.parse.urlparse(file_uri)
    file_uri_path_unquoted = urllib.parse.unquote(file_uri_parsed.path)
    if windows_path and file_uri_path_unquoted.startswith("/"):
        result = path_class(file_uri_path_unquoted[1:])
    else:
        result = path_class(file_uri_path_unquoted)
    if result.is_absolute() == False:
        raise ValueError("Invalid file uri {} : resulting path {} not absolute".format(file_uri, result))
    return result


def print_stats(waveform: Tensor, sample_rate: int = None, src: str = None) -> None:
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    w = waveform.float()
    print(f" - Max:     {w.max().item():6.3f}")
    print(f" - Min:     {w.min().item():6.3f}")
    print(f" - Mean:    {w.mean().item():6.3f}")
    print(f" - Std Dev: {w.std().item():6.3f}")
    print()
    print(waveform)
    print()


def play_audio_jupyter(waveform: Tensor, sample_rate: int) -> Any:
    from IPython.display import Audio, display

    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def play_audio_vscode(waveform: Tensor, sample_rate: int) -> Any:
    # https://github.com/microsoft/vscode-jupyter/issues/1012
    import json

    import IPython.display
    import numpy as np

    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        channels = [waveform.tolist()]
    else:
        channels = waveform.tolist()

    return IPython.display.HTML(
        """
        <script>
            if (!window.audioContext) {
                window.audioContext = new AudioContext();
                window.playAudio = function(audioChannels, sr) {
                    const buffer = audioContext.createBuffer(audioChannels.length, audioChannels[0].length, sr);
                    for (let [channel, data] of audioChannels.entries()) {
                        buffer.copyToChannel(Float32Array.from(data), channel);
                    }
            
                    const source = audioContext.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioContext.destination);
                    source.start();
                }
            }
        </script>
        <button onclick="playAudio(%s, %s)">Play</button>
    """
        % (json.dumps(channels), sample_rate)
    )

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

def zero_grad(model):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()

def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_state(model):
    return sum(s.numel() for s in model.state_dict().values())