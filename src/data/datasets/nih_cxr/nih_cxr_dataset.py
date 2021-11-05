import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import PIL.Image
import click
import torch
from datasets import tqdm
from pytorch_lightning import seed_everything
from skimage import exposure


if __name__ == '__main__':
    datasets_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    root_path = datasets_path.parent.parent
    sys.path.append(str(root_path))


from data.datasets.base_dataset import DetectionDataset, SegmentationDataset
import pandas as pd
import numpy as np

from data.datasets.processing_utils import PIXEL_STATS_FILE_NAME, write_detection_csv_file, \
    compute_detection_dataset_stats, create_segmentation_mask

log = logging.getLogger(__name__)


def prepare_nih_cxr_detection_dataset(path: str, val_percentage: float, test_percentage: float, include_no_findings):
    log.info('Loading targets...')
    annotated_images: List[Tuple[str, list, list]] = parse_labels(path)
    annotated_data = split_by_patients(annotated_images, val_percentage, test_percentage)

    no_findings_data = None
    if include_no_findings > 0:
        no_finding_images = find_no_finding_samples(path, include_no_findings)
        no_findings_data = split_by_patients(no_finding_images, val_percentage, test_percentage)

    train_stats = process_and_write_split(path, 'train', annotated_data, no_findings_data)
    val_stats = process_and_write_split(path, 'validation', annotated_data, no_findings_data)
    test_stats = process_and_write_split(path, 'test', annotated_data, no_findings_data)
    stats = {
        'train': train_stats,
        'validation': val_stats,
        'test': test_stats,
    }
    with open(os.path.join(path, PIXEL_STATS_FILE_NAME), "w") as f:
        json.dump(stats, f)

    log.info('Done')


def process_and_write_split(path: str, split: str, annotated_data, no_findings_data=None):
    patient_ids, samples = annotated_data[split]
    if no_findings_data is not None:
        no_finding_patient_ids, no_finding_samples = no_findings_data[split]
        patient_ids.extend(no_finding_patient_ids)
        samples.extend(no_finding_samples)

    log.info(f'Processing images ({split})...')
    target_img_dir = os.path.join(path, 'processed_images')
    os.makedirs(target_img_dir, exist_ok=True)
    means = []
    vars = []
    all_boxes = []
    for image_id, boxes, _ in tqdm(samples):
        # find image folder
        source_paths = [os.path.join(path, 'images', f'images_{i+1:02d}', 'images', f'{image_id}.png') for i in range(12)]
        source_path_exists = [os.path.exists(source_path) for source_path in source_paths]
        assert sum(source_path_exists) == 1
        source_path = source_paths[source_path_exists.index(True)]
        target_path = os.path.join(target_img_dir, f'{image_id}.png')
        pixel_mean, pixel_var, img_size = process_image(source_path, target_path)
        means.append(pixel_mean)
        vars.append(pixel_var)
        all_boxes.extend(boxes)

    log.info(f'Preparing segmentations masks ({split})...')
    for processed_label_name, processed_label in TARGET_MAPPING.items():
        log.info(f'Label: {processed_label_name}')
        target_folder = os.path.join(path, 'segmentation_masks', processed_label_name)
        os.makedirs(target_folder, exist_ok=True)

        for image_id, boxes, labels in tqdm(samples):
            img_file = os.path.join(target_img_dir, f'{image_id}.png')
            w, h = PIL.Image.open(img_file).size
            mask_file = os.path.join(target_folder, f'{image_id}.gif')
            # filter boxes with other labels
            boxes = [box for box, label in zip(boxes, labels) if label == processed_label]
            create_segmentation_mask(boxes, img_size=(h, w), target_file=mask_file)

    write_detection_csv_file(os.path.join(path, f'{split}.csv'), samples)
    return compute_detection_dataset_stats(patient_ids, means, vars, all_boxes)


TARGET_MAPPING = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 2,
    'Infiltrate': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
}


def find_no_finding_samples(path: str, limit_samples=100) -> List[Tuple[str, list, list]]:
    with open(os.path.join(path, 'test_list.txt')) as file:
        test_image_ids = set([line.rstrip() for line in file.readlines()])
    data_entry_csv = pd.read_csv(os.path.join(path, 'Data_Entry_2017_v2020.csv'))
    data_entry_csv = data_entry_csv[data_entry_csv["Finding Labels"] == 'No Finding']
    data_entry_csv = data_entry_csv[data_entry_csv['Image Index'].isin(test_image_ids)]
    data_entry_csv = data_entry_csv.sample(limit_samples)
    images_ids = data_entry_csv['Image Index'].tolist()
    return [(image_id[:-len('.png')], [], []) for image_id in images_ids]


def parse_labels(path) -> List[Tuple[str, list, list]]:
    bbox_df = pd.read_csv(os.path.join(path, 'BBox_List_2017.csv'))
    extract_box = lambda row: (float(row['Bbox [x']), float(row['y']), float(row['w']), float(row['h]']))
    extract_label = lambda row: TARGET_MAPPING[row['Finding Label']]

    images_dict = {}
    for _, row in bbox_df.iterrows():
        image_id = row['Image Index']
        if image_id not in images_dict:
            images_dict[image_id] = {
                'labels': [],
                'boxes': []
            }

        images_dict[image_id]['labels'].append(extract_label(row))
        images_dict[image_id]['boxes'].append(extract_box(row))

    return [(image_id[:-len('.png')], data['boxes'], data['labels']) for image_id, data in images_dict.items()]


def split_by_patients(samples: List[Tuple[str, list, list]], val_percentage: float, test_percentage: float):
    patients = defaultdict(list)
    for image_id, boxes, labels in samples:
        patient_id = image_id.split('_')[0]  # 00000002_000.png
        assert len(patient_id) == 8
        patient_id = int(patient_id)
        patients[patient_id].append((image_id, boxes, labels))
    patient_ids = list(patients.keys())
    patient_num_samples = [len(patients[patient_id]) for patient_id in patient_ids]
    num_samples = sum(patient_num_samples)
    permuted_patient_indices = np.random.permutation(len(patients))

    val_patient_ids = []
    num_val_samples = 0
    test_patient_ids = []
    num_test_samples = 0
    i = 0
    while num_test_samples < round(test_percentage * num_samples):
        patient_index = permuted_patient_indices[i]
        test_patient_ids.append(patient_ids[patient_index])
        num_test_samples += patient_num_samples[patient_index]
        i += 1
    while num_val_samples < round(val_percentage * num_samples):
        patient_index = permuted_patient_indices[i]
        val_patient_ids.append(patient_ids[patient_index])
        num_val_samples += patient_num_samples[patient_index]
        i += 1
    train_patient_ids = [patient_ids[i] for i in permuted_patient_indices[i:]]

    train_samples = [sample for patient_id in train_patient_ids for sample in patients[patient_id]]
    val_samples = [sample for patient_id in val_patient_ids for sample in patients[patient_id]]
    test_samples = [sample for patient_id in test_patient_ids for sample in patients[patient_id]]

    return {
        'train': (train_patient_ids, train_samples),
        'validation': (val_patient_ids, val_samples),
        'test': (test_patient_ids, test_samples)
    }


def process_image(source_path, target_path):
    img = np.array(PIL.Image.open(source_path).convert('L'), dtype=float) / 255.
    img = exposure.equalize_hist(img)
    pixel_mean = img.mean()
    pixel_var = img.var()
    img_size = img.shape
    img = (255 * img).astype(np.uint8)
    img = PIL.Image.fromarray(img).convert('L')
    img.save(target_path, format='PNG')

    return pixel_mean, pixel_var, img_size


class NihCxrDetectionDataset(DetectionDataset):
    def _get_img_path(self, id):
        return os.path.join(self.root_path, 'processed_images', f'{id}.png')


class NihCxrSegmentationDataset(SegmentationDataset):
    def _get_img_path(self, id):
        return os.path.join(self.root_path, 'processed_images', f'{id}.png')

    def _get_mask_paths(self, id) -> Dict[str, str]:
        return {label: os.path.join(self.root_path, 'segmentation_masks', label, f'{id}.gif')
                for label in TARGET_MAPPING.keys()}


@click.command()
@click.argument('nih_cxr_root_path')
@click.option('--val_percentage', default=0.2)
@click.option('--test_percentage', default=0.2)
@click.option('--include_no_findings', default=100)
@click.option('--seed', default=None)
def prepare_nih_cxr_detection_dataset_cmd(nih_cxr_root_path, val_percentage, test_percentage, include_no_findings, seed):
    if seed is not None:
        seed_everything(seed)
    log.info(f'Preparing NIH CXR dataset...')
    log.info(f'Using path "{nih_cxr_root_path}"')

    prepare_nih_cxr_detection_dataset(nih_cxr_root_path,
                                      val_percentage=val_percentage, test_percentage=test_percentage,
                                      include_no_findings=include_no_findings)
    log.info(f'Finished dataset preparation.')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    prepare_nih_cxr_detection_dataset_cmd()
