import json
import logging
import math
import os
import sys
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List

import PIL
import click
import datasets
import pyarrow
from datasets import features, Split, DatasetInfo
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from pandas import Int64Index, Index
from pyarrow import compute
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset
from tqdm import tqdm

CHEXPERT_LABELS = features.ClassLabel(names=["negative", "positive", "uncertain", "blank"])

if __name__ == '__main__':
    datasets_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    root_path = datasets_path.parent.parent
    sys.path.insert(0, str(root_path))
from data.datasets.processing_utils import PIXEL_STATS_FILE_NAME, compute_pixel_stats

log = logging.getLogger(__name__)


CHEXPERT_FOLDER = 'CheXpert-v1.0-small'


class ChexpertDatasetBuilder(datasets.GeneratorBasedBuilder):
    def __init__(self, *args, **kwargs):
        super(ChexpertDatasetBuilder, self).__init__(*args, **kwargs)

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="chexpert",
            version=datasets.Version("1.0.0", "")
        ),
        datasets.BuilderConfig(
            name="chexpert_ap-pa",
            version=datasets.Version("1.0.0", "")

        )
    ]
    DEFAULT_CONFIG_NAME = "chexpert"

    @property
    def manual_download_instructions(self) -> Optional[str]:
        return None

    def _info(self):
        chexpert_class_labels = CHEXPERT_LABELS
        return DatasetInfo(
            #description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "path": features.Value("string"),
                    "patient_id": features.Value("string"),
                    "patient_study": features.Value("string"),
                    "view": features.Value("string"),
                    "width": features.Value("int32"),
                    "height": features.Value("int32"),
                    "pixel_mean": features.Value("float"),
                    "pixel_var": features.Value("float"),
                    "chexpert_labels": {
                        "Atelectasis": chexpert_class_labels,
                        "Cardiomegaly": chexpert_class_labels,
                        "Consolidation": chexpert_class_labels,
                        "Edema": chexpert_class_labels,
                        "Enlarged Cardiomediastinum": chexpert_class_labels,
                        "Fracture": chexpert_class_labels,
                        "Lung Lesion": chexpert_class_labels,
                        "Lung Opacity": chexpert_class_labels,
                        "Pleural Effusion": chexpert_class_labels,
                        "Pneumonia": chexpert_class_labels,
                        "Pneumothorax": chexpert_class_labels,
                        "Pleural Other": chexpert_class_labels,
                        "Support Devices": chexpert_class_labels,
                        "No Finding": chexpert_class_labels
                    }
                }
            ),
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        root_path = self.config.data_dir

        chexpert_root = os.path.join(root_path, CHEXPERT_FOLDER)

        if self.config.name == 'chexpert_ap-pa':
            frontal_only = True
        else:
            assert self.config.name == 'chexpert'
            frontal_only = False

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "root_path": root_path, "chexpert_root": chexpert_root, "split": "train", "frontal_only": frontal_only
            }),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                "root_path": root_path, "chexpert_root": chexpert_root, "split": "valid", "frontal_only": frontal_only
            }),
        ]

    def _generate_examples(self, root_path: str, chexpert_root: str, split: str, frontal_only: bool):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        log.info(f'========== Creating {split} split ==========')
        log.info(f'-> Reading csv file')
        splits_csv = pd.read_csv(os.path.join(chexpert_root, f'{split}.csv'))
        splits_csv = splits_csv.fillna(value='blank')

        log.info(f'-> Processing samples...')

        for _, sample in splits_csv.iterrows():
            sample = sample.to_dict()
            img_rel_path = str(sample['Path'])
            img_path = os.path.join(root_path, img_rel_path)
            _, split_folder, patient_id, patient_study, file_name = img_rel_path.split('/')
            path = os.path.join(split_folder, patient_id, patient_study, file_name)

            if str(sample['Frontal/Lateral']).strip() == 'Lateral':
                view = 'LA'
                if frontal_only:
                    log.info(f'Skipping lateral sample: {img_path}.')
                    continue
            else:
                view = str(sample['AP/PA']).strip()

            if not os.path.exists(img_path):
                log.warning(f'{img_path} not found. Skipping sample.')
                continue

            img = PIL.Image.open(img_path)
            np_img = np.array(img, dtype=float) / 255.
            pixel_mean = np_img.mean()
            pixel_var = np_img.var()
            return_sample = {
                "path": path,
                "patient_id": patient_id,
                "patient_study": patient_study,
                "view": view,
                "width": img.width,
                "height": img.height,
                "pixel_mean": pixel_mean,
                "pixel_var": pixel_var
            }

            chexpert_columns = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                                "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
                                "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"]

            # ----- chexpert labels -----
            chexpert_label_map = {
                1.0: "positive",
                0.0: "negative",
                -1.0: "uncertain",
                'blank': "blank"
            }

            chexpert_data = {
                key: chexpert_label_map[sample[key]] for key in chexpert_columns
            }

            return_sample["chexpert_labels"] = chexpert_data

            yield img_rel_path, return_sample

    @staticmethod
    def create_dataset(root_path: str, target_path: Optional[str] = None,
                       name: Optional[str] = None, **config_kwargs) -> str:
        """

        :param root_path: This path should contain the folder: Chexpert-v1.0-small
        :param target_path:
        :param name:
        :param config_kwargs:
        :return:
        """
        builder = ChexpertDatasetBuilder(name=name, data_dir=root_path, **config_kwargs)
        builder.download_and_prepare()#download_mode=datasets.GenerateMode.FORCE_REDOWNLOAD)
        dataset: datasets.DatasetDict = builder.as_dataset()

        info: DatasetInfo = dataset["train"].info
        if target_path is None:
            target_path = os.path.join(root_path, f'{info.config_name}_dataset')

        # compute binary labels
        dataset["train"] = map_to_binary(dataset["train"])
        dataset["validation"] = map_to_binary(dataset["validation"])

        val_patient_ids, val_split, train_split = split_by_patients(dataset['train'], min_val_samples=5000)

        # test set is used for validation
        dataset["test"] = dataset["validation"]
        dataset["train"] = train_split
        dataset["validation"] = val_split

        stats = {}
        for split, split_dataset in dataset.items():
            mean, std = compute_pixel_stats(split_dataset, pixel_mean_column='pixel_mean', pixel_var_column='pixel_var')
            pos_weights = compute_chexpert_pos_weights(split_dataset)
            stats[split] = {
                'num_samples': split_dataset.num_rows,
                'scan': {'pixel_mean': mean, 'pixel_std': std},
                'chexpert_bin_pos_weights': pos_weights
            }
        stats["validation"]["patient_ids"] = val_patient_ids

        log.info('Saving dataset...')
        dataset.save_to_disk(target_path)
        with open(os.path.join(target_path, PIXEL_STATS_FILE_NAME), "w") as f:
            json.dump(stats, f)
        os.symlink(os.path.join(root_path, CHEXPERT_FOLDER), os.path.join(target_path, 'images'))

        log.info('Done')

        return target_path


def chexpert_labels():
    return [
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pleural Effusion",
    ]


uncertain_label = CHEXPERT_LABELS.str2int('uncertain')
blank_label = CHEXPERT_LABELS.str2int('blank')


def _map_finding_pos(labels):
    # negative (0.0) => 0
    # positive (1.0) => 1
    labels[labels == uncertain_label] = 1  # uncertain (-1.0)
    labels[labels == blank_label] = 0  # blank ('')
    return labels


def _map_finding_neg(labels):
    # negative (0.0) => 0
    # positive (1.0) => 1
    labels[labels == uncertain_label] = 0  # uncertain (-1.0)
    labels[labels == blank_label] = 0  # blank ('')
    return labels


def map_to_binary(internal_dataset: datasets.Dataset, mode='U_pos'):
    """
    See https://github.com/Stomper10/CheXpert/blob/master/materials.py or https://github.com/jfhealthcare/Chexpert/blob/master/data/dataset.py
    """
    if mode == 'U_pos':
        def map_batch(batch: dict):
            labels = batch['chexpert_labels']
            batch['chexpert_bin_labels'] = {
                "Cardiomegaly": _map_finding_pos(np.array(labels["Cardiomegaly"])),  # index 2 => U_negative
                "Edema": _map_finding_pos(np.array(labels["Edema"])),  # index 5 => U_positive
                "Consolidation": _map_finding_pos(np.array(labels["Consolidation"])),  # index 6 => U_negative
                "Atelectasis": _map_finding_pos(np.array(labels["Atelectasis"])),  # index 8 => U_positive
                "Pleural Effusion": _map_finding_pos(np.array(labels["Pleural Effusion"])),  # index 10 => U_positive ?????
            }
            return batch
    elif mode == 'U_diff':
        def map_batch(batch: dict):
            labels = batch['chexpert_labels']
            batch['chexpert_bin_labels'] = {
                "Cardiomegaly": _map_finding_neg(np.array(labels["Cardiomegaly"])),  # index 2 => U_negative
                "Edema": _map_finding_pos(np.array(labels["Edema"])),  # index 5 => U_positive
                "Consolidation": _map_finding_neg(np.array(labels["Consolidation"])),  # index 6 => U_negative
                "Atelectasis": _map_finding_pos(np.array(labels["Atelectasis"])),  # index 8 => U_positive
                "Pleural Effusion": _map_finding_neg(np.array(labels["Pleural Effusion"])),  # index 10 => U_positive ?????
            }
            return batch
    else:
        raise ValueError(mode)
    return internal_dataset.map(map_batch)


def compute_chexpert_pos_weights(internal_dataset: datasets.Dataset):
    all_columns = set(internal_dataset.data.column_names)
    all_columns.remove('chexpert_bin_labels')
    data = internal_dataset.data
    if internal_dataset._indices is not None:
        data = data.fast_gather(internal_dataset._indices.column(0).to_pylist())
    chexpert_bin_labels = data.drop(all_columns).flatten().to_pandas()
    label_pos_frequencies = chexpert_bin_labels.sum(axis=0) / chexpert_bin_labels.count()
    label_neg_frequencies = 1 - label_pos_frequencies  # these will be the pos weights

    return {
        key[len('chexpert_bin_labels.'):]: value
        for key, value in label_neg_frequencies.to_dict().items()
    }


def split_by_patients(dataset: datasets.Dataset, min_val_samples: int = 5000) -> (List[str], datasets.Dataset, datasets.Dataset):
    df: pd.DataFrame = dataset.data.to_pandas()
    df = df.sort_values(by=['patient_id'])
    num_val_samples = 0
    all_val_indices: Optional[Index] = None
    val_patient_ids = []
    for patient_id, indices in df.groupby('patient_id').groups.items():
        if all_val_indices is None:
            all_val_indices = indices
        else:
            all_val_indices = all_val_indices.union(indices)
        val_patient_ids.append(patient_id)
        num_val_samples += len(indices)
        all_val_indices.append(indices)

        if num_val_samples >= min_val_samples:
            break
    assert all_val_indices is not None
    all_train_indices: Index = df.drop(all_val_indices).index

    val_dataset = dataset.select(all_val_indices)
    train_dataset = dataset.select(all_train_indices)

    return val_patient_ids, val_dataset, train_dataset


class ChexpertDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, image_root_path, stats):
        self.dataset = dataset
        self.image_root_path = image_root_path
        self.stats = stats

    @staticmethod
    def load_from_disk(dataset_path: str, image_root_path=None):
        dataset = datasets.load_from_disk(dataset_path)
        assert isinstance(dataset, datasets.DatasetDict)

        # get and check images path
        if image_root_path is None:
            image_root_path = os.path.join(dataset_path, 'images')
        assert os.path.exists(image_root_path), f'Image path {image_root_path} not found'

        with open(os.path.join(dataset_path, PIXEL_STATS_FILE_NAME), "r") as f:
            stats = json.load(f)

        return {
            split: ChexpertDataset(split_dataset, image_root_path, stats[split])
            for split, split_dataset in dataset.items()
        }

    def __getitem__(self, item):
        sample = self.dataset.__getitem__(item)

        chexpert_labels = sample.pop("chexpert_labels")
        chexpert_bin_labels = sample.pop("chexpert_bin_labels")
        image_path = os.path.join(self.image_root_path, sample['path'])
        scan = PIL.Image.open(image_path)

        return {
            "scan": scan,
            "chexpert_labels": chexpert_labels,
            "chexpert_bin_labels": chexpert_bin_labels
        }

    def __len__(self):
        return len(self.dataset)


@click.command()
@click.argument('chexpert_root_path')
@click.option('--config', default=None)
@click.option('--target_path', default=None)
def create_chexpert_dataset(chexpert_root_path, config, target_path):
    log.info(f'Creating Chexpert dataset ({config})...')
    log.info(f'Using Chexpert root path "{chexpert_root_path}"')
    target_path = ChexpertDatasetBuilder.create_dataset(chexpert_root_path, name=config, target_path=target_path)
    log.info(f'Finished dataset creation. Dataset stored at "{target_path}"')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    create_chexpert_dataset()


