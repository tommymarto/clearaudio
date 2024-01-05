from pathlib import Path

from omegaconf.dictconfig import DictConfig
from hydra import compose, initialize_config_dir


def get_config_path() -> str:
    return str(Path(__file__).parent.parent.resolve() / "clearaudio" / "conf")


def get_samples_path() -> Path:
    return Path(__file__).parent.parent.resolve() / "samples"


def get_test_collated_path() -> Path:
    return get_samples_path() / "test_collated.csv"


def get_test_config() -> DictConfig:
    sample_path = get_samples_path()
    hq = sample_path / "hq"
    lq = sample_path / "lq"
    with initialize_config_dir(config_dir=get_config_path(), job_name="test_app"):
        cfg = compose(config_name="config", overrides=["dataset=base_dataset", "trainer=base_trainer"])
        cfg.dataset.high_quality_path = str(hq)
        cfg.dataset.low_quality_path = str(lq)
        cfg.dataset.collated_path = str(get_test_collated_path())
    return cfg
