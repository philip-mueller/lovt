import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict

import torch
from skimage import exposure

import PIL.Image
import click
import numpy as np
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset
from tqdm import tqdm


if __name__ == '__main__':
    datasets_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    root_path = datasets_path.parent.parent
    sys.path.insert(0, str(root_path))

from data.datasets.base_dataset import SegmentationDataset
from data.datasets.processing_utils import PIXEL_STATS_FILE_NAME

log = logging.getLogger(__name__)


def prepare_covid_rural_dataset(path: str, val_percentage: float, test_percentage: float):
    assert val_percentage + test_percentage < 1.

    patient_scans, means, vars, pos_weights = find_and_process_samples(path)
    patient_ids = list(patient_scans.keys())
    patient_num_samples = [len(patient_scans[patient_id]) for patient_id in patient_ids]
    num_samples = sum(patient_num_samples)

    log.info(f'Splitting dataset and computing stats...')
    permuted_patient_indices = np.random.permutation(len(patient_ids))

    val_patient_indices = []
    num_val_samples = 0
    test_patient_indices = []
    num_test_samples = 0

    i = 0
    while num_test_samples < round(test_percentage * num_samples):
        patient_index = permuted_patient_indices[i]
        test_patient_indices.append(patient_index)
        num_test_samples += patient_num_samples[patient_index]
        i += 1
    while num_val_samples < round(val_percentage * num_samples):
        patient_index = permuted_patient_indices[i]
        val_patient_indices.append(patient_index)
        num_val_samples += patient_num_samples[patient_index]
        i += 1
    train_patient_indices = permuted_patient_indices[i:]

    train_scan_ids, train_stats = get_split(train_patient_indices, patient_ids, patient_scans, means, vars, pos_weights)
    val_scan_ids, val_stats = get_split(val_patient_indices, patient_ids, patient_scans, means, vars, pos_weights)
    test_scan_ids, test_stats = get_split(test_patient_indices, patient_ids, patient_scans, means, vars, pos_weights)

    # compute the dataset stats
    log.info(f'Computing stats...')
    stats = {
        'train': train_stats,
        'validation': val_stats,
        'test': test_stats,
    }
    with open(os.path.join(path, PIXEL_STATS_FILE_NAME), "w") as f:
        json.dump(stats, f)

    # save the ids in CSV files
    log.info(f'Writing csv files...')
    write_csv_file(os.path.join(path, 'train.csv'), train_scan_ids)
    write_csv_file(os.path.join(path, 'validation.csv'), val_scan_ids)
    write_csv_file(os.path.join(path, 'test.csv'), test_scan_ids)


def get_split(train_patient_indices, patient_ids, patient_scans, means, vars, pos_weights):
    train_patient_ids = [patient_ids[i] for i in train_patient_indices]
    train_scan_ids = [scan_id for patient_id in train_patient_ids for scan_id in patient_scans[patient_id]]
    train_means = [mean for patient_id in train_patient_ids for mean in means[patient_id]]
    train_vars = [var for patient_id in train_patient_ids for var in vars[patient_id]]
    train_pos_weights = [pos_weight for patient_id in train_patient_ids for pos_weight in pos_weights[patient_id]]
    train_stats = compute_stats(train_means, train_vars, train_pos_weights, train_patient_ids)
    return train_scan_ids, train_stats


def write_csv_file(target_path, ids):
    with open(target_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID'])
        for id in ids:
            writer.writerow([id])


def compute_stats(means, vars, pos_weights, patient_ids):
    mean = np.mean(means)
    var = np.mean(vars) + np.var(means)

    return {'num_samples': len(means), 'scan': {'pixel_mean': float(mean), 'pixel_std': math.sqrt(float(var))},
            'segmentation_pos_weights': {'opacity': float(np.mean(pos_weights))},
            'patient_ids': patient_ids}


def find_and_process_samples(path: str):
    patients = defaultdict(list)
    missing_masks = []
    masks_dir = os.path.join(path, 'pngs_masks')
    original_scans_dir = os.path.join(path, 'jpgs')
    processed_scans_dir = os.path.join(path, 'processed_images')
    os.makedirs(processed_scans_dir, exist_ok=True)
    pos_weights = defaultdict(list)
    means = defaultdict(list)
    vars = defaultdict(list)
    log.info(f'Processing images...')
    for file in tqdm(os.listdir(original_scans_dir)):
        scan_path = os.path.join(original_scans_dir, file)
        if not os.path.isfile(scan_path) or not file.endswith('.jpg'):
            continue
        assert file.startswith('COVID-19-AR-')

        img_name = file[:-len('.jpg')]
        patient_id = img_name.split('_')[0]

        mean, var, img_size = process_image(scan_path, os.path.join(processed_scans_dir, file))

        mask_path = os.path.join(masks_dir, f'{img_name}.png')
        if os.path.exists(mask_path) and os.path.isfile(mask_path):
            pos_weights[patient_id].append(check_mask_and_compute_pos_weights(mask_path, img_size))
            patients[patient_id].append(img_name)
            means[patient_id].append(mean)
            vars[patient_id].append(var)
        else:
            log.warning(f'Mask not found: {mask_path}')
            missing_masks.append(img_name)

    return patients, means, vars, pos_weights


def process_image(source_path, target_path):
    img = np.array(PIL.Image.open(source_path), dtype=float) / 255.
    img = np.mean(img, axis=-1)  # images are saved as color => convert to greyscale
    img = exposure.equalize_hist(img)
    pixel_mean = img.mean()
    pixel_var = img.var()
    img_size = img.shape
    img = (255 * img).astype(np.uint8)
    img = PIL.Image.fromarray(img).convert('L')
    img.save(target_path, format='JPEG')

    return pixel_mean, pixel_var, img_size


def check_mask_and_compute_pos_weights(path, scan_size):
    mask = torch.as_tensor(np.array(PIL.Image.open(path)), dtype=torch.int64)
    assert mask.shape == scan_size, f'Mask size was {mask.shape} but scan size was {scan_size}'
    values = np.unique(mask)
    assert all(val in (0., 1.) for val in values), values
    return 1 - mask.float().mean()


class CovidRuralDataset(SegmentationDataset):
    def _get_img_path(self, id):
        return os.path.join(self.root_path, 'processed_images', f'{id}.jpg')

    def _get_mask_paths(self, id) -> Dict[str, str]:
        return {'opacity': os.path.join(self.root_path, 'pngs_masks', f'{id}.png')}


@click.command()
@click.argument('covid_rural_root_path')
@click.option('--val_percentage', default=0.2)
@click.option('--test_percentage', default=0.2)
@click.option('--seed', default=None)
def prepare_covid_rural_dataset_cmd(covid_rural_root_path, val_percentage, test_percentage, seed):
    if seed is not None:
        seed_everything(seed)
    log.info(f'Preparing COVID rural dataset...')
    log.info(f'Using path "{covid_rural_root_path}"')

    prepare_covid_rural_dataset(covid_rural_root_path, val_percentage=val_percentage, test_percentage=test_percentage)
    log.info(f'Finished dataset preparation.')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    prepare_covid_rural_dataset_cmd()
