import logging
import hydra
import submitit
import submitit.core.utils

from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from clearaudio.trainers import trainer_wavenet, trainer_vqvae, trainer_diffusion, base_trainer
from clearaudio.utils.utils import get_output_dir

LOG = logging.getLogger(__name__)


@record
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    if cfg.trainer.type == "wavenet":
        trainer = trainer_wavenet.WavenetTrainer(cfg)
    elif cfg.trainer.type == 'vqvae':
        trainer = trainer_vqvae.VQVAE_Trainer(cfg)
    elif cfg.trainer.type == 'diffusion':
        trainer = trainer_diffusion.DiffusionTrainer(cfg)
    elif cfg.trainer.type == 'dummy':
        trainer = base_trainer.DummyTrainer(cfg)
    else:
        raise NotImplementedError

    if cfg.trainer.platform == "local":
        LOG.info(f"Output directory {cfg.trainer.output_dir}/{cfg.trainer.sync_key}")
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
        timeout_min=int(cfg.trainer.slurm.timeout) * 60,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=cfg.trainer.slurm.partition,
        slurm_qos=cfg.trainer.slurm.qos,
        slurm_gres=f"gpu:{cfg.trainer.slurm.gpus_per_node}"
        # slurm_signal_delay_s=120,
        # **kwargs
    )

    executor.update_parameters(name=cfg.trainer.name)

    slurm_additional_parameters = {
        'requeue': True
    }

    if cfg.trainer.slurm.account:
        slurm_additional_parameters['account'] = cfg.trainer.slurm.account

    if cfg.trainer.slurm.reservation:
        slurm_additional_parameters['reservation'] = cfg.trainer.slurm.reservation

    executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)
    try:
        job = executor.submit(trainer)
    except submitit.core.utils.FailedJobError as e:
        a = str(e)
        if a.find('reservation') != -1:  # Invalid reservation in error message
            LOG.warning(f"No active reservation for: {cfg.trainer.slurm.reservation}")
            del slurm_additional_parameters['reservation']
            executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)
            job = executor.submit(trainer)

    LOG.info(f"Submitted job_id: {job.job_id}")
    return job


if __name__ == "__main__":
    main()
