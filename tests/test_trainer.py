from os import confstr
from pathlib import Path

import numpy as np
import pandas as pd

from hydra import compose, initialize_config_dir

from clearaudio.trainers import base_trainer
from clearaudio.datasets import audio

from tests import utils


def get_dataset(subset: str = None) -> audio.AudioDataset:
    cfg = utils.get_test_config()
    dataset = audio.AudioDataset(cfg, subset)

    collated_path = utils.get_test_collated_path()
    if not collated_path.exists():
        dataset.build_collated()
    return dataset


# def test_base_trainer() -> None:
#     with initialize_config_dir(config_dir=utils.get_config_path(), job_name="test_app"):
#         cfg = compose(config_name="config", overrides=["dataset=base_dataset", "trainer=base_trainer"])
#         trainer = base_trainer.BaseTrainer(cfg)
    
def test_load_checkpoint() -> None:
    trainer = None
    with initialize_config_dir(config_dir=utils.get_config_path(), job_name="test_app"):
        cfg = compose(config_name="config", overrides=["dataset=base_dataset", "trainer=base_trainer"])
        trainer = base_trainer.BaseTrainer(cfg)
    checkpoint_path = None
    assert trainer.checkpoint_load(checkpoint_path) is None
    checkpoint_path = "test_42"

    assert trainer.checkpoint_load(checkpoint_path) is None

    checkpoint_path = Path(cfg.trainer.results_dir) / 'checkpoint_epoch_0.pt'
    trainer.checkpoint_dump(checkpoint_path)
    assert trainer.checkpoint_load(checkpoint_path) is not None


def test_find_latest_checkpoint() -> None:
    trainer = None
    with initialize_config_dir(config_dir=utils.get_config_path(), job_name="test_app"):
        cfg = compose(config_name="config", overrides=["dataset=base_dataset", "trainer=base_trainer"])
        trainer = base_trainer.BaseTrainer(cfg)
    
    results = Path(cfg.trainer.results_dir)
    (results / 'checkpoint_epoch_9.pt').touch()
    (results / 'checkpoint_epoch_100.pt').touch()
    ckp = trainer.find_latest_checkpoint_path(str(results))
    assert ckp == results / 'checkpoint_epoch_100.pt'