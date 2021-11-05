import csv
import json
import logging
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import PIL
import click
import datasets
import torch
from datasets import features
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from tqdm import tqdm

if __name__ == '__main__':
    datasets_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    root_path = datasets_path.parent.parent
    sys.path.append(str(root_path))


from data.datasets.chexpert.chexpert_dataset import compute_chexpert_pos_weights, map_to_binary
from data.datasets.processing_utils import PIXEL_STATS_FILE_NAME, compute_pixel_stats
from data.datasets.mimic_cxr.section_parser import custom_mimic_cxr_rules, section_text
from data.text_utils.sentence_splitting import SentenceSplitter


MIMIC_CXR_2_BASE_URL = 'https://physionet.org/files/mimic-cxr/2.0.0/'
MIMIC_CXR_JPG_2_BASE_URL = 'https://physionet.org/files/mimic-cxr-jpg/2.0.0/'

MIMIC_CXR_JPG_FILES = f'{MIMIC_CXR_JPG_2_BASE_URL}/files/'
MIMIC_CXR_PATIENT_PREFIXES = ['p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19']

log = logging.getLogger(__name__)

_CITATION = """\
Johnson, A., Pollard, T., Mark, R., Berkowitz, S., & Horng, S. (2019). 
MIMIC-CXR Database (version 2.0.0). 
PhysioNet. https://doi.org/10.13026/C2JT1Q.
"""


MIMIC_CXR_FOLDER = 'mimic-cxr_2-0-0'
MIMIC_CXR_JPG_FOLDER = 'mimic-cxr-jpg_2-0-0'


@dataclass
class MimicCxrDatasetConfig(datasets.BuilderConfig):
    views: Optional[List[str]] = None
    sections: List[str] = field(default_factory=lambda: ['findings', 'impression'])
    min_section_tokens: int = 3
    unique_patients: bool = False


@dataclass
class MimicCxrDatasetInfo(datasets.DatasetInfo):
    scan_pixel_mean: Optional[float] = None
    scan_pixel_std: Optional[float] = None


class MimicCxrDatasetBuilder(datasets.GeneratorBasedBuilder):
    def __init__(self, *args, **kwargs):
        super(MimicCxrDatasetBuilder, self).__init__(*args, **kwargs)

    BUILDER_CONFIG_CLASS = MimicCxrDatasetConfig
    BUILDER_CONFIGS = [
        MimicCxrDatasetConfig(
            name="mimic-cxr",
            version=datasets.Version("2.0.0", "")
        ),
        MimicCxrDatasetConfig(
            name="mimic-cxr_ap-pa",
            version=datasets.Version("2.0.0", ""),
            views=['PA', 'AP']
        ),
        MimicCxrDatasetConfig(
            name="mimic-cxr_postero-anterior_findings-only",
            version=datasets.Version("2.0.0", ""),
            sections=['findings'],
            views=['PA']
        ),
        MimicCxrDatasetConfig(
            name="mimic-cxr_antero-posterior_findings-only",
            version=datasets.Version("2.0.0", ""),
            sections=['findings'],
            views=['AP']
        ),
        MimicCxrDatasetConfig(
            name="mimic-cxr_lateral_findings-only",
            version=datasets.Version("2.0.0", ""),
            sections=['findings'],
            views=['LATERAL']
        ),
        MimicCxrDatasetConfig(
            name="mimic-cxr_left-lateral_findings-only",
            version=datasets.Version("2.0.0", ""),
            sections=['findings'],
            views=['LL']
        ),
        MimicCxrDatasetConfig(
            name="mimic-cxr_unique-patients",
            version=datasets.Version("2.0.0", ""),
            unique_patients=True
        )
    ]
    DEFAULT_CONFIG_NAME = "mimic-cxr"

    @property
    def manual_download_instructions(self) -> Optional[str]:
        return None

    def _info(self):
        chexpert_class_labels = features.ClassLabel(names=["negative", "positive", "uncertain", "blank"])
        return MimicCxrDatasetInfo(
            #description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "patient_id": features.Value("string"),
                    "study_id": features.Value("string"),
                    "study_date": features.Value("int32"),
                    "scan": features.Sequence({
                        "dicom_id": features.Value("string"),
                        "view": features.Value("string"),
                        "view_name": features.Value("string"),
                        "width": features.Value("int32"),
                        "height": features.Value("int32"),
                        "pixel_mean": features.Value("float"),
                        "pixel_var": features.Value("float"),
                    }),
                    "num_scans": features.Value("int32"),
                    "scan_pixel_mean": features.Value("float"),
                    "scan_pixel_var": features.Value("float"),
                    "report": {
                        "full_text": features.Value("string"),
                        "findings_text": features.Value("string"),
                        "impression_text": features.Value("string"),
                        "text": features.Value("string"),
                        "findings_sentences": features.Sequence(features.Value("string")),
                        "impression_sentences": features.Sequence(features.Value("string")),
                        "sentences": features.Sequence(features.Value("string")),
                        "num_findings_sentences": features.Value("int32"),
                        "num_impression_sentences": features.Value("int32"),
                        "num_sentences": features.Value("int32"),
                        "num_findings_tokens": features.Value("int32"),
                        "num_impression_tokens": features.Value("int32"),
                        "num_tokens": features.Value("int32")
                    },
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
                        "No Finding": features.ClassLabel(names=["blank", "positive"])
                    }
                }
            ),
            supervised_keys=None,
            homepage="https://physionet.org/content/mimic-cxr/2.0.0/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        # download paths:
        # - https://physionet.org/files/mimic-cxr-jpg/2.0.0/
        #     - files
        #     - LICENSE.txt
        #     - README
        #     - SHA256SUMS.txt
        #     - mimic-cxr-2.0.0-chexpert.csv.gz
        #     - mimic-cxr-2.0.0-metadata.csv.gz
        #     - mimic-cxr-2.0.0-negbio.csv.gz
        #     - mimic-cxr-2.0.0-split.csv.gz
        # - https://physionet.org/files/mimic-cxr/2.0.0/
        #     - LICENSE.txt
        #     - SHA256SUMS.txt
        #     - cxr-record-list.csv.gz
        #     - cxr-study-list.csv.gz
        #     - mimic-cxr-reports.zip
        mimic_cxr_root_path = self.config.data_dir

        mimic_cxr_root = os.path.join(mimic_cxr_root_path, MIMIC_CXR_FOLDER)
        mimic_cxr_jpg_root = os.path.join(mimic_cxr_root_path, MIMIC_CXR_JPG_FOLDER)

        log.info('Extracting reports...')
        reports_folder = os.path.join(mimic_cxr_root, 'mimic-cxr-reports')
        if not os.path.exists(reports_folder):
            with zipfile.ZipFile(os.path.join(mimic_cxr_root, 'mimic-cxr-reports.zip'), "r") as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    try:
                        zf.extract(member, reports_folder)
                    except zipfile.error as e:
                        pass
        else:
            log.info('Already extracted, skipping extraction.')

        splits_file = os.path.join(mimic_cxr_jpg_root, 'mimic-cxr-2.0.0-split.csv.gz')
        metadata_file = os.path.join(mimic_cxr_jpg_root, 'mimic-cxr-2.0.0-metadata.csv.gz')
        chexpert_file = os.path.join(mimic_cxr_jpg_root, 'mimic-cxr-2.0.0-chexpert.csv.gz')
        reports_folder = os.path.join(reports_folder, 'files')
        images_folder = os.path.join(mimic_cxr_jpg_root, 'files')
        assert os.path.exists(images_folder), f'Image folder {images_folder} not found'

        kwargs = {
            'reports_folder': reports_folder,
            'images_folder': images_folder,
            'splits_file': splits_file,
            'metadata_file': metadata_file,
            'chexpert_file': chexpert_file
        }

        log.info('Initializing SentenceSplitter...')
        self.report_processor = SentenceSplitter(lang='en',
                                                 section_splitter=split_into_sections,
                                                 optional_sections=['impression'])

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={**kwargs, "split": "train"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={**kwargs, "split": "validate"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={**kwargs, "split": "test"}),
        ]

    def _generate_examples(self, reports_folder: str, images_folder: str,
                           splits_file: str, metadata_file: str, chexpert_file: str,
                           split: str):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        log.info(f'========== Creating {split} split ==========')
        log.info(f'-> Reading csv files')
        splits_csv = pd.read_csv(splits_file)
        splits_csv = splits_csv.set_index(['subject_id', 'study_id', 'dicom_id'])
        meta_csv = pd.read_csv(metadata_file)
        meta_csv = meta_csv.set_index(['subject_id', 'study_id', 'dicom_id'])
        chexpert_csv = pd.read_csv(chexpert_file)
        chexpert_csv = chexpert_csv.fillna(value='blank').set_index(['subject_id', 'study_id'])

        splits_csv = splits_csv[splits_csv["split"] == split]
        splits_csv = splits_csv.join(meta_csv)

        if self.config.views is not None:
            log.info(f'-> Filtering views')
            splits_csv = splits_csv[splits_csv["ViewPosition"].isin(self.config.views)]

        if self.config.unique_patients:
            log.info(f'-> Filtering non-unique patients')
            splits_csv = splits_csv.groupby("subject_id").first().reset_index()

        log.info(f'-> Processing samples...')

        prev_subject_prefix = None
        for (subject_id, study_id), group in splits_csv.groupby(["subject_id", "study_id"]):

            subject_id = str(subject_id)
            if prev_subject_prefix != subject_id[:2]:
                prev_subject_prefix = subject_id[:2]
                log.info(f'----- Processing subjects p{prev_subject_prefix} -----')

            study_id = str(study_id)
            study_subpath = os.path.join(f'p{subject_id[:2]}', f'p{subject_id}', f's{study_id}')

            # ----- report -----
            txt_path = os.path.join(reports_folder, f'{study_subpath}.txt')
            with open(txt_path, encoding="utf-8") as f:
                full_report_text = f.read()

            processed_report = self.report_processor(full_report_text, study=f's{study_id}')
            if processed_report is None:
                log.warning(f'Relevant report sections not found for p{subject_id}/s{study_id}. Skipping sample.')
                continue
            findings_text, findings_sentences, findings_tokens = processed_report['findings']
            impression_text, impression_sentences, impression_tokens = processed_report['impression']

            assert 1 <= len(self.config.sections) <= 2
            if 'findings' in self.config.sections and 'impression' in self.config.sections:
                sentences = findings_sentences + impression_sentences
                text = findings_text + ' ' + impression_text
                num_tokens = findings_tokens + impression_tokens
            else:
                assert len(self.config.sections) == 1
                if 'findings' in self.config.sections:
                    sentences = findings_sentences
                    text = findings_text
                    num_tokens = findings_tokens
                elif 'impression' in self.config.sections:
                    sentences = impression_sentences
                    text = impression_text
                    num_tokens = impression_tokens
                else:
                    raise ValueError(self.config.sections)

            if len(sentences) == 0:
                log.warning(f'Relevant report sections ({self.config.sections}) not found for p{subject_id}/s{study_id}. '
                         f'Skipping sample.')
                continue
            if num_tokens < self.config.min_section_tokens:
                log.warning(f'Number of tokens ({num_tokens}) below threshold ({self.config.min_section_tokens}) '
                         f'for p{subject_id}/s{study_id}. Skipping sample.')
                continue

            report = {
                    "full_text": full_report_text,
                    "findings_text": findings_text,
                    "impression_text": impression_text,
                    "text": text,
                    "findings_sentences": findings_sentences,
                    "impression_sentences": impression_sentences,
                    "sentences": sentences,
                    "num_findings_sentences": len(findings_sentences),
                    "num_impression_sentences": len(impression_sentences),
                    "num_sentences": len(sentences),
                    "num_findings_tokens": findings_tokens,
                    "num_impression_tokens": impression_tokens,
                    "num_tokens": num_tokens
            }

            # ----- scan views -----
            views = []
            view_means = []
            view_vars = []
            study_date = None
            for (_, _, dicom_id), view in group.iterrows():
                dicom_id = str(dicom_id)
                study_date = int(view['StudyDate'])

                img_path = os.path.join(images_folder, study_subpath, f'{dicom_id}.jpg')
                if not os.path.exists(img_path):
                    log.warning(f'{img_path} not found. Skipping view.')
                    continue
                img = PIL.Image.open(img_path)
                np_img = np.array(img, dtype=float) / 255.
                pixel_mean = np_img.mean()
                pixel_var = np_img.var()
                view_means.append(pixel_mean)
                view_vars.append(pixel_var)
                views.append({"dicom_id": dicom_id,
                              "view": str(view["ViewPosition"]),
                              "view_name": str(view["ViewCodeSequence_CodeMeaning"]),
                              "width": img.width,
                              "height": img.height,
                              "pixel_mean": pixel_mean,
                              "pixel_var": pixel_var})
            if len(views) == 0:
                log.info(f'No views found for p{subject_id}/s{study_id}. Skipping sample.')
                continue

            # aggregate means and variance over all views
            scan_pixel_mean = np.mean(view_means)
            scan_pixel_var = np.mean(view_vars) + np.var(view_means)

            # ----- chexpert labels -----
            chexpert_data = chexpert_csv.loc[int(subject_id), int(study_id)].to_dict()
            chexpert_label_map = {
                1.0: "positive",
                0.0: "negative",
                -1.0: "uncertain",
                'blank': "blank"
            }

            chexpert_data = {
                key: chexpert_label_map[label] for key, label in chexpert_data.items()
            }

            yield study_id, {
                "study_id": study_id,
                "patient_id": subject_id,
                "study_date": study_date,
                "scan": views,
                "num_scans": len(views),
                "scan_pixel_mean": scan_pixel_mean,
                "scan_pixel_var": scan_pixel_var,
                "report": report,
                "chexpert_labels": chexpert_data
            }

    @staticmethod
    def create_dataset(mimic_cxr_root_path: str, target_path: Optional[str] = None,
                       name: Optional[str] = None, **config_kwargs) -> str:
        """

        :param mimic_cxr_root_path: This path should contain two folders:
            - mimic-cxr_2-0-0: containing the MIMIC-CXR dataset, the "files" folder (i.e. the images) is not required
            - mimic-cxr-jpg_2-0-0: containing the full MIMIC-CXR-JPG dataset
        :param target_path:
        :param name:
        :param config_kwargs:
        :return:
        """
        builder = MimicCxrDatasetBuilder(name=name, data_dir=mimic_cxr_root_path, **config_kwargs)
        builder.download_and_prepare() #download_mode=datasets.GenerateMode.FORCE_REDOWNLOAD)
        dataset: datasets.DatasetDict = builder.as_dataset()

        info: MimicCxrDatasetInfo = dataset["train"].info
        if target_path is None:
            target_path = os.path.join(mimic_cxr_root_path, f'{info.config_name}_dataset')
        assert not os.path.exists(target_path)

        # compute binary labels
        dataset["train"] = map_to_binary(dataset["train"])
        dataset["validation"] = map_to_binary(dataset["validation"])
        dataset["test"] = map_to_binary(dataset["test"])

        save_dataset(dataset, os.path.join(mimic_cxr_root_path, MIMIC_CXR_JPG_FOLDER, 'files'), target_path)

        log.info('Done')

        return target_path


def save_dataset(dataset, images_path, target_path):
    stats = {}
    for split, split_dataset in dataset.items():
        mean, std = compute_pixel_stats(split_dataset)
        pos_weights = compute_chexpert_pos_weights(split_dataset)
        stats[split] = {'scan': {'pixel_mean': mean, 'pixel_std': std}, 'chexpert_bin_pos_weights': pos_weights}
    log.info('Saving dataset...')
    dataset.save_to_disk(target_path)
    with open(os.path.join(target_path, PIXEL_STATS_FILE_NAME), "w") as f:
        json.dump(stats, f)
    os.symlink(images_path,
               os.path.join(target_path, 'images'))


def create_subdataset(source_path: str, target_path: str, views: List[str]=None, sections: List[str]=None, min_tokens=3, superpixel_path=None) -> str:
    def remove_scans_without_superpixels(sample: dict):
        # problem: p12/p12590117/s55245526/1d0bafd0-72c92e4c-addb1c57-40008638-b9ec8584.jpg
        patient_id = f'p{sample["patient_id"]}'
        patient_prefix = patient_id[:3]
        study_id = f's{sample["study_id"]}'
        scan = sample['scan']

        indices = []
        view_removed = False
        for i, dicom_id in enumerate(scan['dicom_id']):
            mask = PIL.Image.open(os.path.join(superpixel_path, patient_prefix, patient_id, study_id, f'{dicom_id}.png'))
            if np.sum(mask) == 0:
                log.info(f'Skipping sample because it has no superpixels: {patient_prefix}/{patient_id}/{study_id}/{dicom_id}')
                view_removed = True
                continue
            else:
                indices.append(i)
        if view_removed:
            scan['dicom_id'] = [scan['dicom_id'][i] for i in indices]
            scan['view'] = [scan['view'][i] for i in indices]
            scan['view_name'] = [scan['view_name'][i] for i in indices]
            scan['width'] = [scan['width'][i] for i in indices]
            scan['height'] = [scan['height'][i] for i in indices]
            scan['pixel_mean'] = [scan['pixel_mean'][i] for i in indices]
            scan['pixel_var'] = [scan['pixel_var'][i] for i in indices]
            sample['scan'] = scan
        return sample

    def remove_views(scan: dict):
        indices = [i for i, view in enumerate(scan['view']) if view in views]
        scan['dicom_id'] = [scan['dicom_id'][i] for i in indices]
        scan['view'] = [scan['view'][i] for i in indices]
        scan['view_name'] = [scan['view_name'][i] for i in indices]
        scan['width'] = [scan['width'][i] for i in indices]
        scan['height'] = [scan['height'][i] for i in indices]
        scan['pixel_mean'] = [scan['pixel_mean'][i] for i in indices]
        scan['pixel_var'] = [scan['pixel_var'][i] for i in indices]

        return scan

    def remove_sections(report: dict):
        if 'findings' in sections and 'impression' in sections:
            report['sentences'] = report['findings_sentences'] + report['impression_sentences']
            report['text'] = report['findings_text'] + ' ' + report['impression_text']
            report['num_tokens'] = report['num_findings_tokens'] + report['num_impression_tokens']
        else:
            if 'findings' in sections:
                report['sentences'] = report['findings_sentences']
                report['text'] = report['findings_text']
                report['num_tokens'] = report['num_findings_tokens']
            elif 'impression' in sections:
                report['sentences'] = report['impression_sentences']
                report['text'] = report['impression_text']
                report['num_tokens'] = report['num_impression_tokens']
            else:
                raise ValueError(sections)
        report['num_sentences'] = len(report['sentences'])

        return report

    def remove_views_and_sections(sample: dict) -> dict:
        sample['scan'] = remove_views(sample['scan'])
        # aggregate means and variance over all views
        sample['scan_pixel_mean'] = np.mean(sample['scan']['pixel_mean'])
        sample['scan_pixel_var'] = np.mean(sample['scan']['pixel_varn']) + np.var(sample['scan']['pixel_mean'])
        sample['num_scans'] = len(sample['scan']['dicom_id'])
        sample['report'] = remove_sections(sample['report'])

        return sample

    def filter_samples_without_views_and_text(sample):
        return len(sample['scan']['dicom_id']) > 0 and sample['report']['num_tokens'] >= min_tokens

    dataset = datasets.load_from_disk(source_path)
    assert isinstance(dataset, datasets.DatasetDict)
    if views is not None or sections is not None:
        dataset = dataset.map(remove_views_and_sections)
    if superpixel_path is not None:
        dataset = dataset.map(remove_scans_without_superpixels)
    dataset = dataset.filter(filter_samples_without_views_and_text)
    images_path = os.readlink(os.path.join(source_path, 'images'))
    save_dataset(dataset, images_path, target_path)
    return target_path


class MimicCxrDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, image_root_path, stats, superpixel_path, mask_path):
        self.dataset = dataset
        self.image_root_path = image_root_path
        self.stats = stats
        self.superpixel_path = superpixel_path
        self.mask_path = mask_path

    @staticmethod
    def load_from_disk(dataset_path: str, image_root_path=None, superpixel_path=None, mask_path=None):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # load dataset (for all columns except for images)
        dataset = datasets.load_from_disk(dataset_path)
        assert isinstance(dataset, datasets.DatasetDict)

        # get and check images path
        if image_root_path is None:
            image_root_path = os.path.join(dataset_path, 'images')
        assert os.path.exists(image_root_path), f'Image path {image_root_path} not found'
        for patient_prefix in ('p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19'):
            img_path = os.path.join(image_root_path, patient_prefix)
            assert os.path.exists(img_path), \
                f'Image path for patients {patient_prefix} not found: {img_path}'
        if superpixel_path is not None:
            assert os.path.exists(superpixel_path)
        if mask_path is not None:
            assert os.path.exists(mask_path)

        # load stats
        with open(os.path.join(dataset_path, PIXEL_STATS_FILE_NAME), "r") as f:
            stats = json.load(f)

        return {
            split: MimicCxrDataset(split_dataset, image_root_path, stats[split], superpixel_path, mask_path)
            for split, split_dataset in dataset.items()
        }

    def __getitem__(self, item):
        sample = self.dataset.__getitem__(item)
        patient_id = f'p{sample["patient_id"]}'
        patient_prefix = patient_id[:3]
        study_id = f's{sample["study_id"]}'
        scan = sample['scan']
        images = []
        for dicom_id in scan['dicom_id']:
            image_path = os.path.join(self.image_root_path, patient_prefix, patient_id, study_id, f'{dicom_id}.jpg')
            images.append(PIL.Image.open(image_path))
        scan['scan'] = images

        if self.superpixel_path is not None:
            superpixel_masks = []
            for dicom_id in scan['dicom_id']:
                mask_path = os.path.join(self.superpixel_path, patient_prefix, patient_id, study_id, f'{dicom_id}.png')
                superpixel_masks.append(PIL.Image.open(mask_path))
            scan['superpixel_mask'] = superpixel_masks
        if self.mask_path is not None:
            masks = []
            for dicom_id in scan['dicom_id']:
                mask_path = os.path.join(self.mask_path, patient_prefix, patient_id, study_id, f'{dicom_id}.png')
                masks.append(PIL.Image.open(mask_path))
            scan['mask'] = masks
        return sample

    def __len__(self):
        return len(self.dataset)


def split_into_sections(full_text_report: str, study: str):
    # exclude these special cases
    custom_section_names, custom_indices = custom_mimic_cxr_rules()
    if study in custom_indices or study in custom_section_names:
        return None

    sections, section_names, section_idx = section_text(full_text_report)
    if 'findings' not in section_names and 'impression' not in section_names:
        return None

    sections_by_name = {'impression': '', 'findings': ''}  # default is the empty impression section
    for section, name in zip(sections, section_names):
        if name in ('findings', 'impression'):
            sections_by_name[name] = section

    return sections_by_name


def create_image_dataset(dataset_path):
    dataset = datasets.load_from_disk(dataset_path)
    assert isinstance(dataset, datasets.DatasetDict)
    for split, split_dataset in dataset.items():
        log.info(f'Processing {split} split...')
        target_csv_file = os.path.join(dataset_path, split, 'image_list.csv')
        assert not os.path.exists(target_csv_file)
        with open(target_csv_file, 'w', newline='') as csvfile:
            header = ['patient_id', 'study_id', 'dicom_id', 'img_file',
                      'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for sample in tqdm(split_dataset):
                for dicom_id in sample['scan']['dicom_id']:
                    writer.writerow({
                        'patient_id': sample['patient_id'],
                        'study_id': sample['study_id'],
                        'dicom_id': dicom_id,
                        'img_file': os.path.join(f'p{sample["patient_id"][:2]}',
                                                 f'p{sample["patient_id"]}',
                                                 f's{sample["study_id"]}',
                                                 f'{dicom_id}.jpg'),
                        'Cardiomegaly': sample['chexpert_bin_labels']['Cardiomegaly'],
                        'Edema': sample['chexpert_bin_labels']['Edema'],
                        'Consolidation': sample['chexpert_bin_labels']['Consolidation'],
                        'Atelectasis': sample['chexpert_bin_labels']['Atelectasis'],
                        'Pleural Effusion': sample['chexpert_bin_labels']['Pleural Effusion']
                    })


class MimicCxrImageDataset(Dataset):
    def __init__(self, dataset_csv_file, image_root_path, stats):
        super(MimicCxrImageDataset, self).__init__()
        self.image_root_path = image_root_path
        self.stats = stats

        self.img_paths = []
        self.chexpert_labels = []
        self.label_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
        with open(dataset_csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for sample in reader:
                self.img_paths.append(sample['img_file'])
                self.chexpert_labels.append(tuple(sample[label_name] for label_name in self.label_names))

    @staticmethod
    def load_from_disk(dataset_path: str, image_root_path=None):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # get and check images path
        if image_root_path is None:
            image_root_path = os.path.join(dataset_path, 'images')
        assert os.path.exists(image_root_path), f'Image path {image_root_path} not found'
        for patient_prefix in ('p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19'):
            img_path = os.path.join(image_root_path, patient_prefix)
            assert os.path.exists(img_path), \
                f'Image path for patients {patient_prefix} not found: {img_path}'

        # load stats
        with open(os.path.join(dataset_path, PIXEL_STATS_FILE_NAME), "r") as f:
            stats = json.load(f)

        return {
            split: MimicCxrImageDataset(os.path.join(dataset_path, split, 'image_list.csv'), image_root_path, split_stats)
            for split, split_stats in stats.items()
        }

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img_path = os.path.join(self.image_root_path, img_path)
        img = PIL.Image.open(img_path)

        labels = self.chexpert_labels[item]
        target = {label_name: torch.tensor(int(label), dtype=torch.int64) for label_name, label in zip(self.label_names, labels)}

        return {'scan': img, 'target': target}

    def __len__(self):
        return len(self.img_paths)


@click.group()
def cli():
    pass


@cli.command('create')
@click.argument('mimic_cxr_root_path')
@click.option('--config', default=None)
@click.option('--target_path', default=None)
def create_mimic_cxr_dataset(mimic_cxr_root_path, config, target_path):
    log.info(f'Creating MIMIC-CXR dataset ({config})...')
    log.info(f'Using MIMIC-CXR root path "{mimic_cxr_root_path}"')
    target_path = MimicCxrDatasetBuilder.create_dataset(mimic_cxr_root_path, name=config, target_path=target_path)
    log.info(f'Finished dataset creation. Dataset stored at "{target_path}"')


@cli.command('create_image_list')
@click.option('--path')
def create_image_list_cmd(path):
    log.info(f'Creating image lists for MIMIC-CXR dataset "{path}"...')
    create_image_dataset(path)
    log.info('Done.')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    cli()
