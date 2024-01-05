from os import confstr
from pathlib import Path

import numpy as np
import pandas as pd

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from clearaudio.datasets import audio
from tests import utils


def get_dataset(subset: str = None) -> audio.AudioDataset:
    cfg = utils.get_test_config()
    with open_dict(cfg):
        cfg.dataset.match_sample_rate = False
        cfg.dataset.use_collated_file = False
    dataset = audio.AudioDataset(cfg, subset)
    dataset.build_collated()

    # collated_path = utils.get_test_collated_path()
    # if not collated_path.exists():
        
    return dataset


def test_extra_dataset() -> None:
    #  context initialization
    with initialize_config_dir(config_dir=utils.get_config_path(), job_name="test_app"):
        cfg = compose(config_name="config", overrides=["dataset=extra_dataset", "trainer=base_trainer"])
        assert cfg.dataset.name == "extra"


def test_train_val_test_split() -> None:
    files = list(range(100))
    rng = np.random.default_rng(seed=42)
    train, validate, test = audio.split_dataset(files, 0.5, 0.2, rng)
    assert len(train) == 50
    assert len(validate) == 20
    assert len(test) == 30


def test_split_dataframe() -> None:
    df = pd.DataFrame(np.random.randint(0,100, size=(100, 1)), columns=list('A'))
    rng = np.random.default_rng(seed=42)
    df = audio.split_dataframe(df, 0.5, 0.2, rng)
    assert len(df[df.subset == 'train']) == 50
    assert len(df[df.subset == 'validate']) == 20
    assert len(df[df.subset == 'test']) == 30


def test_load_audio_clip() -> None:
    invalid_audio = utils.get_samples_path() / "hq" / "skip" / "empty.wav"
    clip = audio.load_audio_clip(invalid_audio)
    assert clip is None

    starwars = utils.get_samples_path() / "hq" / "starwars.wav"
    clip = audio.load_audio_clip(starwars)
    assert clip.sample_rate == 22050
    assert clip.num_frames == 22050 * 60

    clip = audio.load_audio_clip(starwars, frame_offset=1000, num_frames=1000)
    assert clip.sample_rate == 22050
    assert clip.num_frames == 1000


def test_find_audio_files() -> None:
    sample_path = utils.get_samples_path()

    hq = sample_path / "hq"
    skip = sample_path / "hq" / "skip"
    df = audio.find_audio_files(hq, "wav", skip)
    assert len(df) >= 1

    # Force to find 48k samples
    df = audio.find_audio_files(hq, "wav", skip, 48000)
    assert df is None


def test_match_lq_files() -> None:
    sample_path = utils.get_samples_path()
    cfg = utils.get_test_config()

    hq = sample_path / "hq"
    skip = sample_path / "hq" / "skip"
    df = audio.find_audio_files(hq, "wav", skip)
    df.rename(columns={"path": "hq_path"}, inplace=True)
    df = audio.match_lq_files(df, cfg)
    assert df is not None

    lq2 = sample_path / "lq2"
    cfg.dataset.low_quality_path = str(lq2)
    df = audio.match_lq_files(df, cfg)
    assert df is None


def test_get_sample_mono_resample() -> None:
    sample = utils.get_samples_path() / "hq" / "sample1.wav"
    cfg = utils.get_test_config()
    cfg.dataset.match_sample_rate = False
    cfg.dataset.mono = False
    cfg.dataset.sample_rate = 44100
    dataset = audio.AudioDataset(cfg)

    clip = dataset.get_sample(sample)
    assert clip.sample_rate == 44100
    assert clip.num_channels == 2

    # Force resample and mono
    cfg.dataset.match_sample_rate = True
    cfg.dataset.mono = True
    cfg.dataset.sample_rate = 22050

    dataset = audio.AudioDataset(cfg)
    clip = dataset.get_sample(sample)
    assert clip.sample_rate == 22050
    assert clip.num_channels == 1

    clip = dataset.get_sample(sample, 0, 44100)
    assert clip.num_frames == 44100
    assert clip.sample_rate == 22050


def test_dataset_build_collated() -> None:
    cfg = utils.get_test_config()

    collated_path = utils.get_test_collated_path()
    if collated_path.exists():
        collated_path.unlink()

    # No collated file
    dataset = audio.AudioDataset(cfg)
    assert dataset.df is None

    # Create it
    dataset.build_collated()
    assert collated_path.exists()

    # Read from file
    dataset = audio.AudioDataset(cfg)
    assert dataset.df is not None
    assert 'sample_rate' in dataset.df.columns


def test_change_subset() -> None:
    dataset = get_dataset()
    assert dataset.chosen_set == None

    dataset.change_subset('train')
    assert len(dataset) > 0
    assert dataset.subset_view.iloc[0]['subset'] == 'train'

    # Overshooting
    clip = dataset.get_hq_sample(100)
    assert clip is None

    clip = dataset.get_hq_sample(0)
    assert clip is not None

    dataset = get_dataset('test')
    assert len(dataset) > 0
    assert dataset.subset_view.iloc[0]['subset'] == 'test'


def test_random_hq_sample() -> None:
    dataset = get_dataset()

    # overshooting
    clip = dataset.get_random_hq_sample(10000, 100000)
    assert clip is None

    # Sample length too long
    whole_clip = dataset.get_hq_sample(0)
    clip = dataset.get_random_hq_sample(0, 100000)

    assert whole_clip.num_frames == clip.num_frames

    clip = dataset.get_random_hq_sample(0, 10)
    clip2 = dataset.get_random_hq_sample(0, 10)

    assert clip.num_frames == clip2.num_frames
    assert clip.frame_offset != clip2.frame_offset


def test_lq_sample() -> None:
    dataset = get_dataset()

    idx = dataset.df[dataset.df.name == 'sample1'].index[0]
    #out of bounds
    clip = dataset.get_lq_sample(1000, eq='eq_user')
    assert clip is None

    # invalid eq name
    clip = dataset.get_lq_sample(0, eq='eq1000')
    assert clip is None

    # valid
    clip = dataset.get_lq_sample(idx, eq='eq_user')
    assert clip is not None

    # eq from file
    clip = dataset.get_lq_sample(idx, eq='eq1')
    assert clip is not None

    # Generate EQ
    idx = dataset.df[dataset.df.name == 'starwars'].index[0]
    clip = dataset.get_lq_sample(idx, eq='eq1')
    assert clip is not None

    # Generate EQ and save EQ2
    idx = dataset.df[dataset.df.name == 'starwars'].index[0]
    clip = dataset.get_lq_sample(idx, eq='eq2', persist_eq=True)
    assert clip is not None

    sample_path = utils.get_samples_path()
    lq_folder = sample_path / "lq"
    sw_path = lq_folder / 'starwars_eq2.wav'
    assert sw_path.exists()
    # remove it
    sw_path.unlink()


def test_get_random_clips() -> None:
    dataset = get_dataset()

    # user eq file
    dataset.cfg.dataset.use_user_lq_files = True
    idx = dataset.df[dataset.df.name == 'sample1'].index[0]
    user_lq, hq, eq_name = dataset.get_random_clips(idx)

    assert user_lq.name == 'sample1' + dataset.cfg.dataset.user_lq_suffix + '.wav'

    # use generate eqs
    dataset.cfg.dataset.use_user_lq_files = False
    lq, hq2, eq_name = dataset.get_random_clips(idx)
    assert lq.name != 'sample1' + dataset.cfg.dataset.user_lq_suffix + '.wav'

    # Test for different random samples
    assert hq.frame_offset != hq2.frame_offset


def test_dataloader() -> None:
    dataset = get_dataset()
    loader = audio.create_dataloader(dataset, None, max_workers=0)
    for epoch in range(2):
        for idx, sample_pair in enumerate(loader):
            # only get the first pair
            if idx > 0:
                continue
            (lq, hq, eq) = sample_pair
            # print(lq)
            assert lq.is_pinned()
            assert hq.is_pinned()

    loader = audio.create_dataloader(dataset, "train", max_workers=20)
    for idx, sample_pair in enumerate(loader):
        # only get the first pair
        if idx > 0:
            continue
        (lq, hq, eq_name) = sample_pair
        # print(lq)
        assert lq.is_pinned()
