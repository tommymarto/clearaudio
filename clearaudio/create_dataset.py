import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from clearaudio.datasets import audio


LOG = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    LOG.info(f"Dataset path: {cfg.dataset.high_quality_path}")
    dataset = audio.AudioDataset(cfg)
    if dataset.should_build_dataset():
        dataset.build_collated()
    LOG.info(f"Whole dataset: {len(dataset)} {cfg.dataset.audio_format.upper()} files!")

    dataset.change_subset("train")
    LOG.info(f"Training dataset: {len(dataset)} {cfg.dataset.audio_format.upper()} files!")

    dataset.change_subset("validate")
    LOG.info(f"Validation dataset: {len(dataset)} {cfg.dataset.audio_format.upper()} files!")

    dataset.change_subset("test")
    LOG.info(f"Testing dataset: {len(dataset)} {cfg.dataset.audio_format.upper()} files!")


if __name__ == "__main__":
    main()
