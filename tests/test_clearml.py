from clearml import Task

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tests import utils


# PUT VPN
# def test_simple_task() -> None:
#     task = Task.init(project_name='ClearAudio', task_name='Simple Task test runner')
#     task.close()


# def test_log_hydra() -> None:
#     with initialize_config_dir(
#         config_dir=utils.get_config_path(), job_name="test_app"
#     ):
#
# cfg = compose(config_name="config", overrides=["dataset=base_dataset", 'trainer=base_trainer'])
#         task = Task.init(project_name='ClearAudio', task_name='Hydra Task test runner')
#         # logger = task.get_logger()
#         task.connect(OmegaConf.to_object(cfg))
#         # print(OmegaConf.to_yaml(cfg))

#         Task.current_task().upload_artifact('hydra config', artifact_object=OmegaConf.to_yaml(cfg))
#         task.close()
