#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

# Dataset Dir Tree:
# ROOT:
# |-- annotation-instance-segmentation
# |   |-- metadata
# |   |   |-- challenge-2019-classes-description-segmentable.csv  # class_csv for short.
# |   |   |-- challenge-2019-label300-segmentable-hierarchy.json  # hierarchy_json for short
# |   |-- train
# |   |   |-- challenge-2019-train-masks
# |   |   |   |-- challenge-2019-train-segmentation-masks.csv     # mask_csv for short.
# |   |   |   |-- challenge-2019-train-masks-[0~f].zip
# |   |   |-- all-masks                                           # N_MASK: 2_125_530
# |   |   |-- challenge-2019-train-segmentation-bbox.csv          # bbox_csv for short.
# |   |   |-- challenge-2019-train-segmentation-labels.csv        # label_csv for short.
# |   |-- validation
# |       |-- challenge-2019-validation-masks
# |       |   |-- challenge-2019-validation-segmentation-masks.csv
# |       |   |-- challenge-2019-validation-masks-[0~f].zip
# |       |-- all-masks                                           # N_MASK: 23_366
# |       |-- challenge-2019-validation-segmentation-bbox.csv
# |       |-- challenge-2019-validation-segmentation-labels.csv
# |-- train       # N_IMAGE: 1_743_042
# |-- validation  # N_IMAGE:    41_620
# |-- test        # N_IMAGE:   125_436

import argparse
import glob
import os
import urllib.parse
import urllib.request
import tarfile
from zipfile import ZipFile

CONFIG = {
    'download_dir': '/home/tyushang_gmail_com/jupyter/datasets/oid/',
    'tarfile_path': None,
}
ABS_DOWNLOAD_DIR = os.path.abspath(CONFIG['download_dir'])
ABS_TAR_PATH     = os.path.abspath(CONFIG['tarfile_path']) if CONFIG.get('tarfile_path') else None

DOWNLOAD_DESC = {
    # images for train/validation/test
    'images'                                   : None,
    # Object Detection track anno
    'annotation-object-detection'              : None,
    # Instance Segmentation track anno
    'annotation-instance-segmentation'         : {
        'train'     : {
            'challenge-2019-train-segmentation-labels.csv': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-segmentation-labels.csv',
            'challenge-2019-train-segmentation-bbox.csv'  : 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-segmentation-bbox.csv',
            'challenge-2019-train-masks'                  : [
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-0.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-1.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-2.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-3.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-4.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-5.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-6.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-7.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-8.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-9.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-a.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-b.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-c.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-d.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-e.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/train-masks/challenge-2019-train-masks-f.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-segmentation-masks.csv',
            ]
        },
        'validation': {
            'challenge-2019-validation-segmentation-labels.csv': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-segmentation-labels.csv',
            'challenge-2019-validation-segmentation-bbox.csv'  : 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-segmentation-bbox.csv',
            'challenge-2019-validation-masks'                  : [
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-0.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-1.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-2.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-3.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-4.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-5.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-6.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-7.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-8.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-9.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-a.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-b.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-c.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-d.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-e.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/validation-masks/challenge-2019-validation-masks-f.zip',
                'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-segmentation-masks.csv',
            ],
        },
        'metadata'  : {
            'challenge-2019-classes-description-segmentable.csv': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-classes-description-segmentable.csv',
            'challenge-2019-label300-segmentable-hierarchy.json': 'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-label300-segmentable-hierarchy.json',
        },
    },
    # Visual Relationships Detection track anno
    'annotation-visual-relationships-detection': None,
}


def is_url(s: str):
    return s.startswith('https://')


def url_filename(s: str):
    res = urllib.parse.urlparse(s)
    return res.path.split('/')[-1]


def download(desc):
    def _download(k, v):
        if v is None:
            return
        elif type(v) == dict:
            os.makedirs(k, exist_ok=True)
            os.chdir(k)
            for k2, v2 in v.items():
                _download(k2, v2)
            os.chdir('..')
        elif type(v) == list:
            os.makedirs(k, exist_ok=True)
            os.chdir(k)
            for e in v:
                urllib.request.urlretrieve(e, url_filename(e))
            os.chdir('..')
        elif is_url(v):
            print(f'download {v}')
            urllib.request.urlretrieve(v, k)

    pwd = os.getcwd()
    try:
        if not os.path.exists(CONFIG['download_dir']):
            os.makedirs(CONFIG['download_dir'])
        os.chdir(CONFIG['download_dir'])
        for k, v in desc.items():
            _download(k, v)
    finally:
        os.chdir(pwd)


def unzip_masks(zip_file_path, extract_dir):
    # if not os.path.exists(extract_dir):
    #     os.makedirs(all_masks_dir)
    print(f'unzip {zip_file_path} -> {extract_dir}')
    with ZipFile(zip_file_path, 'r') as zf:
        zf.extractall(extract_dir)


def create_tar(dir_to_tar, abs_tar_path):
    basename = os.path.basename(abs_tar_path)
    os.chdir(os.path.dirname(abs_tar_path))
    with tarfile.open(basename, 'w') as tar:
        tar.add(dir_to_tar, arcname=basename.split('.')[0])


if __name__ == '__main__':
    # use hard-coded CONFIG if it defined, else, use CLI.
    if 'CONFIG' not in dir():
        parser = argparse.ArgumentParser(description='Set download dir, and tarfile path(optional).'
                                                     'files to be downloaded, url and path-tree '
                                                     'are hard-coded by DOWNLOAD_DESC.')
        parser.add_argument('--download-dir', required=True, default='./oid', help='Set download dir.')
        parser.add_argument('--tarfile-path', help='Set tarfile(optional), if not set, do not tar.')
        args = parser.parse_args()
        CONFIG = args.__dict__

    download(DOWNLOAD_DESC)

    # unzip all the mask files.
    for tv in ['train', 'validation']:
        all_masks_dir  = CONFIG['download_dir'] + f'annotation-instance-segmentation/{tv}/all-masks'
        zip_file_paths = glob.glob(CONFIG['download_dir'] +
                                   f'annotation-instance-segmentation/{tv}/challenge-2019-{tv}-masks/*.zip')
        for zfp in zip_file_paths:
            unzip_masks(zfp, all_masks_dir)

    # for kaggle server, we need to tar all the downloaded files.
    if ABS_TAR_PATH is not None:
        create_tar(ABS_DOWNLOAD_DIR, ABS_TAR_PATH)

