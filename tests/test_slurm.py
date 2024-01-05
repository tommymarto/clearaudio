from os import confstr
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from clearaudio.trainers import base_trainer
from clearaudio.datasets import audio

from clearaudio.train import main

import time 

from tests import utils

##TODO
def test_slurm_loading_checkpoint() -> None:
    trainer = None
    with initialize_config_dir(config_dir=utils.get_config_path(), job_name="test_app"):
        cfg = compose(config_name="config", overrides=["dataset=maestro_scitas", "trainer=base_trainer"])
    with open_dict(cfg):
        cfg.trainer.platform = "slurm"

    job = main(cfg)

    while job.state != "RUNNING":
         time.sleep(1) 
    time.sleep(1)
    job._interrupt(timeout=True)
    job = main(cfg)


    


        

