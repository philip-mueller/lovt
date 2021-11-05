# https://www.kaggle.com/peterchang77/exploratory-data-analysis
# https://github.com/j-bd/yolo_v3/blob/master/pneumonia_functions.py
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import PIL
import click
import pandas as pd

import pydicom
from pytorch_lightning import seed_everything
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    datasets_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    root_path = datasets_path.parent.parent
    sys.path.append(str(root_path))


from data.datasets.base_dataset import DetectionDataset, SegmentationDataset
from data.datasets.processing_utils import PIXEL_STATS_FILE_NAME, write_detection_csv_file, \
    compute_detection_dataset_stats, create_segmentation_mask, random_split_indices

log = logging.getLogger(__name__)


def prepare_rsna_pneunomia_dataset(path: str, val_percentage: float, test_percentage: float):
    assert val_percentage + test_percentage < 1.

    log.info('Loading targets')
    patients: List[Tuple[str, List[Tuple[float, float, float, float]], List[int]]] = parse_labels(path)
    pids: List[str] = [pid for pid, boxes, classes in patients]

    log.info('Processing images')
    src_img_folder = os.path.join(path, 'stage_2_train_images')
    target_img_folder = os.path.join(path, 'stage_2_train_images_processed')
    os.makedirs(target_img_folder, exist_ok=True)
    means = []
    vars = []
    for pid in tqdm(pids):
        mean, var = process_image(src_img_folder, target_img_folder, pid)
        means.append(mean)
        vars.append(var)

    prepare_segmentation_masks(patients, os.path.join(path, 'stage_2_train_masks'), target_img_folder)

    num_samples = len(patients)
    train_indices, val_indices, test_indices = random_split_indices(num_samples, test_percentage, val_percentage)

    log.info('Processing Splits...')
    train_stats = write_split(os.path.join(path, 'train.csv'), train_indices, patients, means, vars)
    val_stats = write_split(os.path.join(path, 'validation.csv'), val_indices, patients, means, vars)
    test_stats = write_split(os.path.join(path, 'test.csv'), test_indices, patients, means, vars)
    stats = {
        'train': train_stats,
        'validation': val_stats,
        'test': test_stats,
    }
    with open(os.path.join(path, PIXEL_STATS_FILE_NAME), "w") as f:
        json.dump(stats, f)
    log.info('Done.')


def write_split(target_path, indices, patients, means, vars):
    patients = [patients[i] for i in indices]
    means = [means[i] for i in indices]
    vars = [vars[i] for i in indices]
    pids = [pid for pid, boxes, classes in patients]
    all_boxes = [box for pid, boxes, classes in patients for box in boxes]

    stats = compute_detection_dataset_stats(pids, means, vars, all_boxes)
    write_detection_csv_file(target_path, patients)
    return stats


def parse_labels(path):
    """
    Based on https://www.kaggle.com/peterchang77/exploratory-data-analysis
    :param path:
    :return:
    """
    labels_df = pd.read_csv(os.path.join(path, 'stage_2_train_labels.csv'))

    extract_box = lambda row: (float(row['x']), float(row['y']), float(row['width']), float(row['height']))

    patients_dict = {}
    for _, row in labels_df.iterrows():
        pid = row['patientId']
        if pid not in patients_dict:
            patients_dict[pid] = {
                'label': row['Target'],
                'boxes': []
            }

        # --- Add box if opacity is present
        if patients_dict[pid]['label'] == 1:
            patients_dict[pid]['boxes'].append(extract_box(row))
    patients_list = []
    for pid, data in patients_dict.items():
        label = int(data['label'])
        boxes = data['boxes']
        box_classes = [label for _ in range(len(boxes))]
        patients_list.append((pid, boxes, box_classes))
    return patients_list


def process_image(src_folder: str, target_folder: str, pid):
    src_file = os.path.join(src_folder, f'{pid}.dcm')
    target_file = os.path.join(target_folder, f'{pid}.png')

    dcm_data = pydicom.read_file(src_file)
    im = PIL.Image.fromarray(dcm_data.pixel_array)
    im.save(target_file, format='PNG')

    np_img = np.array(im, dtype=float) / 255.
    pixel_mean = np_img.mean()
    pixel_var = np_img.var()
    return pixel_mean, pixel_var


def prepare_segmentation_masks(patients, target_folder, img_folder):
    os.makedirs(target_folder, exist_ok=True)

    for pid, boxes, _ in patients:
        img_file = os.path.join(img_folder, f'{pid}.png')
        w, h = PIL.Image.open(img_file).size
        mask_file = os.path.join(target_folder, f'{pid}.gif')
        create_segmentation_mask(boxes, img_size=(h, w), target_file=mask_file)


class RsnaPneunomiaDetectionDataset(DetectionDataset):
    def _get_img_path(self, id):
        return os.path.join(self.root_path, 'stage_2_train_images_processed', f'{id}.png')


class RsnaPneunomiaSegmentationDataset(SegmentationDataset):
    def _get_img_path(self, id):
        return os.path.join(self.root_path, 'stage_2_train_images_processed', f'{id}.png')

    def _get_mask_paths(self, id) -> Dict[str, str]:
        return {'opacity': os.path.join(self.root_path, 'stage_2_train_masks', f'{id}.gif')}


@click.command()
@click.argument('rsna_pneunomia_detection_root_path')
@click.option('--val_percentage', default=0.2)
@click.option('--test_percentage', default=0.2)
@click.option('--seed', default=None)
def prepare_rsna_pneunomia_dataset_cmd(rsna_pneunomia_detection_root_path, val_percentage, test_percentage, seed):
    if seed is not None:
        seed_everything(seed)
    log.info(f'Preparing RSNA Pneunomia Detection dataset...')
    log.info(f'Using path "{rsna_pneunomia_detection_root_path}"')

    prepare_rsna_pneunomia_dataset(rsna_pneunomia_detection_root_path, val_percentage=val_percentage, test_percentage=test_percentage)
    log.info(f'Finished dataset preparation.')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    prepare_rsna_pneunomia_dataset_cmd()
