
import logging
import hydra
import submitit
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from clearaudio.trainers import base_trainer
from clearaudio.utils.utils import get_output_dir


LOG = logging.getLogger(__name__)


@record
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    if cfg.trainer.platform == "local":
        LOG.info(f"Output directory {cfg.trainer.output_dir}/{cfg.trainer.sync_key}")
        trainer = base_trainer.DummyTrainer(cfg)
        trainer.setup_platform()
        trainer.setup_trainer()
        trainer.run()
        return 0

    with open_dict(cfg):
        cfg.trainer.output_dir = str(get_output_dir(cfg, cfg.trainer.sync_key))

    # Mode SLURM
    executor = submitit.AutoExecutor(folder=cfg.trainer.output_dir, slurm_max_num_timeout=30)
    executor.update_parameters(
        mem_gb=cfg.trainer.slurm.mem,
        gpus_per_node=cfg.trainer.slurm.gpus_per_node,
        tasks_per_node=cfg.trainer.slurm.gpus_per_node,  # one task per GPU
        cpus_per_task=cfg.trainer.slurm.cpus_per_task,
        nodes=cfg.trainer.slurm.nodes,
        timeout_min=int(cfg.trainer.slurm.timeout), # min instad of hours
        # Below are cluster dependent parameters
        slurm_partition=cfg.trainer.slurm.partition,
        slurm_qos=cfg.trainer.slurm.qos,
        slurm_gres=f"gpu:{cfg.trainer.slurm.gpus_per_node}"
        # slurm_signal_delay_s=120,
        # **kwargs
    )

    executor.update_parameters(name=cfg.trainer.name)
    if cfg.trainer.slurm.account:
        executor.update_parameters(slurm_account=cfg.trainer.slurm.account)

    trainer = base_trainer.DummyTrainer(cfg)
    job = executor.submit(trainer)
    LOG.info(f"Submitted job_id: {job.job_id}")
    return 0


if __name__ == "__main__":
    main()
