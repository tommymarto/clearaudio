import math
from os import confstr
from pathlib import Path

import torch
import torchaudio
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from clearaudio.datasets import audio
from clearaudio.transforms import signal
from tests import utils


def test_cuda() -> None:
    assert torch.cuda.is_available()


def test_resample_audio() -> None:
    hq = utils.get_samples_path() / "hq" / "sample1.wav"
    clip = audio.load_audio_clip(hq)
    resampled, _ = signal.resample_signal(clip.waveform, clip.sample_rate, clip.sample_rate // 4)
    assert math.ceil(clip.waveform.shape[1] / 4) == resampled.shape[1]


def test_load_audio_mono_python() -> None:
    hq = utils.get_samples_path() / "hq" / "sample1.wav"
    clip_stereo = audio.load_audio_clip(hq)
    assert clip_stereo.num_channels == 2
    clip_mono = audio.load_audio_clip(hq, mono=True)
    assert clip_mono.num_channels == 1
    assert clip_stereo.sample_rate == clip_mono.sample_rate


def test_mono_sox() -> None:
    hq = utils.get_samples_path() / "hq" / "sample1.wav"
    clip_stereo = audio.load_audio_clip(hq)
    assert clip_stereo.num_channels == 2

    sox = signal.SoxEffectTransform()
    sox.to_mono()
    mono_sox, mono_sr = sox.apply_file(str(hq))

    assert mono_sox.shape[0] == 1
    assert clip_stereo.sample_rate == mono_sr

    clip_mono = audio.load_audio_clip(hq, mono=True)
    assert torch.equal(clip_mono.waveform, mono_sox)

    lq = utils.get_samples_path() / "lq" / "test_mono_sample1_sox.wav"
    torchaudio.save(str(lq), mono_sox, mono_sr)


def test_read_effect_conf() -> None:
    #  context initialization
    with initialize_config_dir(config_dir=utils.get_config_path(), job_name="test_app"):
        cfg = compose(config_name="config", overrides=["dataset=base_dataset", "trainer=base_trainer"])
        # print(cfg.dataset.low_quality_effect.items())
        transforms = signal.SoxEffectTransform.from_config(cfg)
        assert len(transforms) >= 2


def test_process_file() -> None:
    sox = signal.SoxEffectTransform("test_eq")
    sox.add_equalizer(600, -30, 1)
    sox.add_equalizer(100, -30, 0.7)
    # sox.add_equalizer(5000, 30, 0.7)
    hq = str(utils.get_samples_path() / "hq" / "sample1.wav")
    lq_folder = str(utils.get_samples_path() / "lq")
    output_file = sox.process_file(hq, lq_folder)
    assert Path(output_file).exists()


def test_eq1() -> None:
    #  context initialization
    with initialize_config_dir(config_dir=utils.get_config_path(), job_name="test_app"):
        cfg = compose(config_name="config", overrides=["dataset=base_dataset", "trainer=base_trainer"])
        # print(cfg.dataset.low_quality_effect.items())
        transforms = signal.SoxEffectTransform.from_config(cfg)
        sox = transforms[0]
        hq = str(utils.get_samples_path() / "hq" / "sample1.wav")
        lq_folder = str(utils.get_samples_path() / "lq")
        output_file = sox.process_file(hq, lq_folder, True)
        assert Path(output_file).exists()
