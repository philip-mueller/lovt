import os
from dataclasses import dataclass
from glob import glob
from typing import Optional, List, Tuple

import click
import hydra
import torch
import torch.nn.functional as F
from datasets import tqdm
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from hydra.experimental import compose, initialize
from torch.utils.data import DataLoader

from data.dataloading_utils import DatasetTransformWrapper, load_dataset
from models.image.scan_encoder import ScanInputInfo, ScanEncoderModel, ScanEncoderConfig
from models.pretraining.pretraining_utils import load_encoder
from common.dataclass_utils import TensorDataclassMixin


@dataclass
class LocalScanDownstreamEmbeddings(TensorDataclassMixin):
    x: ScanInputInfo
    yg: torch.Tensor  # (B x d_g)
    yl: torch.Tensor  # (B x N x d_l)

    classes_x: Optional[torch.Tensor] = None  # (B x C x H_x x W_x)
    classes_g: Optional[torch.Tensor] = None  # (B x C)
    class_probs_yl: Optional[torch.Tensor] = None  # (B x C x H x W)
    classes_yl: Optional[torch.Tensor] = None  # (B x C x H x W)
    class_names: Optional[List[str]] = None

    def save(self, folder: str, batch_idx: int):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.to_dict(), os.path.join(folder, f'local_scan_downstream_{batch_idx}.pt'))

    @staticmethod
    def load(folder: str, batch_idx: int, device='cpu'):
        path = os.path.join(folder, f'local_scan_downstream_{batch_idx}.pt')
        if os.path.exists(path):
            data = torch.load(path, map_location=device)
            data['x'] = ScanInputInfo.from_dict(data['x'])
            return LocalScanDownstreamEmbeddings(**data)
        else:
            raise IndexError

    def normalize(self):
        return self.apply(lambda x: F.normalize(x, dim=-1, p=2), ignore=['x', 'classes_x', 'classes_g',
                                                                         'class_probs_yl', 'classes_yl',
                                                                         'class_names'])

    @staticmethod
    def num_batches(folder: str):
        return len(glob(f'{folder}/local_scan_downstream_*.pt'))


class LocalScanDownstreamEmbeddingsExporter(pl.LightningModule):
    def __init__(self, encoder: ScanEncoderModel, segmentation_task, batch_size, num_workers):
        super(LocalScanDownstreamEmbeddingsExporter, self).__init__()
        self.encoder = encoder
        self.batch_size = batch_size
        self.num_workers = num_workers

        if segmentation_task is None:
            self.target_names = None
        elif segmentation_task == 'SCR_hear_lungs_clavicles':
            self.target_names = ['heart', 'lungs', 'clavicles']
        elif segmentation_task == 'rsna_pneunomia_detection':
            self.target_names = ['opacity']
        else:
            raise ValueError(segmentation_task)

    def forward(self, scan: torch.Tensor, segmentation_masks: dict = None, id=None, **kwargs):

        encoded = self.encoder(scan.to(self.device, non_blocking=True))
        (H, W) = encoded.local_structure_size

        if id is None and 'dicom_id' in kwargs:
            id = kwargs['dicom_id']

        if self.target_names is not None:
            assert segmentation_masks is not None
            segmentation_mask = torch.stack([segmentation_masks[target].to(self.device, non_blocking=True)
                                             for target in self.target_names], dim=1)  # (B x C x H_seg x W_seg)
            classes_g = (segmentation_mask > 0).any(dim=-1).any(dim=-1)  # (B x C)
            # (B x C x H x W)
            class_probs_l = F.interpolate(segmentation_mask.float(), size=(H, W), mode='bilinear', align_corners=False)
            classes_l = class_probs_l > 0.5

            return LocalScanDownstreamEmbeddings(
                x=self.encoder.get_x(scan, dicom_id=id, view_index=0, view=None),
                yg=encoded.global_features,
                yl=encoded.local_features,
                classes_x=segmentation_mask,
                classes_g=classes_g,
                class_probs_yl=class_probs_l,
                classes_yl=classes_l,
                class_names=self.target_names
            ).detach()
        else:
            return LocalScanDownstreamEmbeddings(
                x=self.encoder.get_x(scan, dicom_id=id, view_index=0, view=None),
                yg=encoded.global_features,
                yl=encoded.local_features)

    def run_export(self, dataset, data, predictions_dir):
        dataloader = DataLoader(DatasetTransformWrapper(load_dataset(dataset)[data],
                                                        self.encoder.val_transform),
                                batch_size=self.batch_size,
                                num_workers=self.num_workers, pin_memory=True)
        for i, batch in tqdm(enumerate(dataloader)):
            embeddings = self(**batch)
            embeddings.save(predictions_dir, batch_idx=i)

    @staticmethod
    def export_model(model: 'BiModalModelRepresentationLearner', predictions_dir, segmentation_task, dataset,
                     data='test', gpu=None):
        exporter = LocalScanDownstreamEmbeddingsExporter(model.model_a.encoder, segmentation_task,
                                                         batch_size=model.config.batch_size,
                                                         num_workers=model.num_workers)
        if gpu is not None:
            device = torch.device("cuda", gpu)
        else:
            device = torch.device("cpu")
        exporter = exporter.to(device=device)

        exporter.run_export(dataset, data, predictions_dir)

    @staticmethod
    def export_baseline(encoder_config: ScanEncoderConfig, predictions_dir, segmentation_task, dataset,
                        data='test', gpu=None, batch_size=8, num_workers=10):
        exporter = LocalScanDownstreamEmbeddingsExporter(load_encoder(encoder_config, dataset_stats=None),
                                                         segmentation_task,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers)
        if gpu is not None:
            device = torch.device("cuda", gpu)
        else:
            device = torch.device("cpu")
        exporter = exporter.to(device=device)

        exporter.run_export(dataset, data, predictions_dir)
