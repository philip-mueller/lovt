import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Dict, Any

from omegaconf import MISSING, OmegaConf
from torch.utils.data import Dataset, Subset

from data.datasets.COVID_rural.covid_rural_dataset import CovidRuralDataset
from data.datasets.chexpert.chexpert_dataset import ChexpertDataset
from data.datasets.mimic_cxr.mimic_cxr_dataset import MimicCxrDataset, MimicCxrImageDataset
from data.datasets.nih_cxr.nih_cxr_dataset import NihCxrDetectionDataset, NihCxrSegmentationDataset
from data.datasets.object_cxr.object_cxr_dataset import ObjectCxrDetectionDataset, ObjectCxrSegmentationDataset
from data.datasets.rsna_pneunomia_detection.rsna_pneunomia_detection_dataset import RsnaPneunomiaDetectionDataset, \
    RsnaPneunomiaSegmentationDataset
from data.datasets.siim_acr_pneumothorax.siim_acr_pneumothorax import SIIMSegmentationDataset


class DatasetTransformWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


@dataclass
class DatasetConfig:
    name: str = MISSING
    path: str = MISSING
    loader: str = MISSING

    train_subset: Optional[float] = None
    val_subset: Optional[float] = None
    test_subset: Optional[float] = None

    arguments: Dict[str, Any] = field(default_factory=dict)


def load_dataset_config(name: str):
    src_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    dataset_configs_path = src_path.parent.joinpath('configs', 'dataset')
    config_path = dataset_configs_path.joinpath(f'{name}.yaml')
    return OmegaConf.merge(DatasetConfig, OmegaConf.load(config_path))


DATASET_LOADERS = {
    'MimicCxrDataset': MimicCxrDataset,
    'MimicCxrImageDataset': MimicCxrImageDataset,
    'ChexpertDataset': ChexpertDataset,
    'CovidRuralDataset': CovidRuralDataset,
    'RsnaPneunomiaDetectionDataset': RsnaPneunomiaDetectionDataset,
    'RsnaPneunomiaSegmentationDataset': RsnaPneunomiaSegmentationDataset,
    'NihCxrDetectionDataset': NihCxrDetectionDataset,
    'NihCxrSegmentationDataset': NihCxrSegmentationDataset,
    'SIIMSegmentationDataset': SIIMSegmentationDataset,
    'ObjectCxrDetectionDataset': ObjectCxrDetectionDataset,
    'ObjectCxrSegmentationDataset': ObjectCxrSegmentationDataset
}


def load_dataset(config: Union[str, DatasetConfig]):
    if isinstance(config, str):
        config = load_dataset_config(config)
    if config.loader not in DATASET_LOADERS:
        raise ValueError(config.loader)
    datasets = DATASET_LOADERS[config.loader].load_from_disk(config.path, **config.arguments)

    datasets['train'] = apply_subset(datasets['train'], config.train_subset)
    datasets['validation'] = apply_subset(datasets['validation'], config.val_subset)
    datasets['test'] = apply_subset(datasets['test'], config.test_subset)

    return datasets


def apply_subset(dataset, subset: Optional[float]):
    if subset is None:
        return dataset
    else:
        num_samples = round(len(dataset) * subset)
        indices = list(range(num_samples))
        sub_dataset = Subset(dataset, indices)
        if hasattr(dataset, 'stats'):
            sub_dataset.stats = dataset.stats
        return sub_dataset
