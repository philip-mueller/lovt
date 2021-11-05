import logging
import os
import shutil
from dataclasses import dataclass
from typing import Tuple, Optional, Collection

import torch
from omegaconf import MISSING
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models import ResNet
from torchvision.transforms import RandomApply
from torchvision.transforms import RandomApply
import torchvision.transforms.functional as F
from data.dataloading_utils import load_dataset
from common.script_utils import BaseExperimentConfig
from common.user_config import USER_CONFIG
from models.image.scan_transforms import PadToSquare, Resize

log = logging.getLogger(__name__)

def load_backbone(config):
    repo_or_dir, model = config.backbone_model
    if config.backbone_checkpoint is not None:
        config.backbone_pretrained = False
    model = torch.hub.load(repo_or_dir, model, pretrained=config.backbone_pretrained)
    assert config.backbone_architecture == 'resnet'
    assert isinstance(model, ResNet)
    return model


@dataclass
class BaselineModelConfig:
    backbone_architecture: str = MISSING
    backbone_model: Tuple[str, str] = (MISSING, MISSING)
    backbone_checkpoint: Optional[str] = None
    backbone_pretrained: bool = False
    input_size: Collection[int] = (MISSING, MISSING)

    batch_size: int = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    max_epochs: int = MISSING
    warmup_epochs: int = 0

    dataset: str = MISSING


@dataclass
class BaselineExperimentConfig(BaseExperimentConfig):
    model_config: BaselineModelConfig = MISSING


def export_pretrained_weights(config, model, trainer, last_model=False, model_extraction_fn=None):
    backbone_weights_path = os.path.join(trainer.checkpoint_callback.dirpath, 'backbone_weights.pt')
    log.info(f"----- Extracting backbone weights ({backbone_weights_path}) -----")
    model_path = trainer.checkpoint_callback.last_model_path if last_model else trainer.checkpoint_callback.best_model_path
    loaded_model: nn.Module = model.load_from_checkpoint(model_path)
    if model_extraction_fn is None:
        backbone_model = loaded_model.model
    else:
        backbone_model = model_extraction_fn(loaded_model)
    torch.save(backbone_model.state_dict(), backbone_weights_path)
    log.info(f'Backbone weights saved to {backbone_weights_path}')

    run_path = os.getcwd()
    os.chdir('..')
    target_path = os.path.join(USER_CONFIG.models.base_path, 'baselines', config.name)
    assert not os.path.exists(target_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.move(run_path, target_path)
    os.chdir(target_path)
    log.info(f'Moved model folder from {run_path} to {target_path}')


def prepare_transform(config, mode):
    dataset_stats = load_dataset(config.dataset)['train'].stats
    pixel_mean = dataset_stats.get('pixel_mean', 0.5)
    pixel_std = dataset_stats.get('pixel_std', 0.5)
    img_size_1, img_size_2 = config.input_size
    assert img_size_1 == img_size_2
    kernel_size = int(0.1 * img_size_1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # note greyscale images => no color distortions (saturation/hue jittering), solarization or random greyscale
    if mode == 'PixelPro':
        transform = nn.Sequential(
            Resize((img_size_1, img_size_1)),
            T.RandomApply([T.ColorJitter(0.8, 0.8, saturation=0, hue=0)], p=0.8),
            T.RandomApply([T.GaussianBlur((kernel_size, kernel_size), (0.1, 2.0))], p=0.5),
            T.Normalize(mean=pixel_mean, std=pixel_std)
        )
    elif mode == 'SimCLR':
        transform = torch.nn.Sequential(
            T.RandomResizedCrop((img_size_1, img_size_1)),
            T.RandomApply(
                [T.ColorJitter(0.8, 0.8, saturation=0, hue=0)],  p=0.8),  # greyscale images => no color jittering
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur((kernel_size, kernel_size), (0.1, 2.0))], p=0.5),
            T.Normalize(mean=pixel_mean, std=pixel_std)
        )
    elif mode == 'VICReg':
        transform = torch.nn.Sequential(
            T.RandomResizedCrop((img_size_1, img_size_1), scale=(0.2, 1.0)),
            T.RandomApply(
                [T.ColorJitter(0.4, 0.4, saturation=0, hue=0)],  p=0.8),  # greyscale images => no color jittering
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur((kernel_size, kernel_size), (0.1, 2.0))], p=0.5),
            T.Normalize(mean=pixel_mean, std=pixel_std)
        )
    else:
        raise ValueError(mode)
    return img_size_1, transform


class DatasetImageOnlyWrapper(Dataset):
    def __init__(self, dataset: Dataset, max_size=512) -> None:
        self.dataset = dataset
        self.max_size = max_size

    def __getitem__(self, idx):
        scan = F.resize(self.dataset[idx]['scan'], (self.max_size, self.max_size))
        scan = F.to_tensor(scan)
        return scan.expand(3, -1, -1)

    def __len__(self):
        return len(self.dataset)


class TwoImageTransformsWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform, max_size=512) -> None:
        self.dataset = dataset
        self.transform = transform
        self.max_size = max_size

    def __getitem__(self, idx):
        scan = F.resize(self.dataset[idx]['scan'], (self.max_size, self.max_size))
        scan = F.to_tensor(scan)
        scan = scan.expand(3, -1, -1)
        return self.transform(scan), self.transform(scan)

    def __len__(self):
        return len(self.dataset)

