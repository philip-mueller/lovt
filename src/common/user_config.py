import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING, OmegaConf


@dataclass
class WandbConfig:
    user: str = MISSING
    project: str = MISSING


@dataclass
class ModelsFolderConfig:
    base_path: str = MISSING
    pretrained_models_folder: str = MISSING


@dataclass
class UserConfig:
    wandb: WandbConfig = MISSING
    models: ModelsFolderConfig = MISSING


def load_user_config():
    src_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    user_config_path = src_path.parent.joinpath('configs', 'user_config.yml')
    return OmegaConf.load(user_config_path)


USER_CONFIG: UserConfig = load_user_config()
