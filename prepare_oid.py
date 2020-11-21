#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"


import copy
import os
import pickle
import platform
import sys
from typing import *

import numpy as np
import pandas as pd
import torch
from cv2 import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetFromList, MapDataset, build_batch_data_loader
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data.detection_utils import build_augmentation
from detectron2.data.samplers import TrainingSampler
from detectron2.engine import default_argument_parser, default_setup
from detectron2.structures import BoxMode, Instances, Boxes, BitMasks
from tqdm import tqdm

RUN_ON = 'local' if platform.node() == 'frank-note' else 'google-cloud'

if RUN_ON == 'local':
    DETECTRON2_DIR = '/data/venv-pytorch/detectron2'
    OID_DIR        = '/data/venv-tensorflow2/open-images-dataset'
else:
    DETECTRON2_DIR = '/home/tyushang_gmail_com/jupyter/detectron2'
    OID_DIR        = '/home/jupyter/datasets/oid'

MASK_THRESHOLD = 127


def get_paths(oid_root, tvt):
    """
    @param oid_root:
    @param tvt: 'train', 'validation' or 'test'
    Dataset Dir Tree:
    ROOT:
    |-- annotation-instance-segmentation
    |   |-- metadata
    |   |   |-- challenge-2019-classes-description-segmentable.csv  # class_csv for short.
    |   |   |-- challenge-2019-label300-segmentable-hierarchy.json  # hierarchy_json for short
    |   |-- train
    |   |   |-- challenge-2019-train-masks
    |   |   |   |-- challenge-2019-train-segmentation-masks.csv     # mask_csv for short.
    |   |   |   |-- challenge-2019-train-masks-[0~f].zip
    |   |   |-- all-masks                                           # N_MASK: 2125530
    |   |   |-- challenge-2019-train-segmentation-bbox.csv          # bbox_csv for short.
    |   |   |-- challenge-2019-train-segmentation-labels.csv        # label_csv for short.
    |   |-- validation
    |       |-- challenge-2019-validation-masks
    |       |   |-- challenge-2019-validation-segmentation-masks.csv
    |       |   |-- challenge-2019-validation-masks-[0~f].zip
    |       |-- all-masks                                           # N_MASK: 23366
    |       |-- challenge-2019-validation-segmentation-bbox.csv
    |       |-- challenge-2019-validation-segmentation-labels.csv
    |-- train       # N_IMAGE
    |-- validation  # N_IMAGE
    |-- test        # N_IMAGE
    """
    return {
        'klass_csv' : os.path.join(oid_root, 'annotation-instance-segmentation/metadata/'
                                             'challenge-2019-classes-description-segmentable.csv'),
        'mask_csv'  : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/'
                                             f'challenge-2019-{tvt}-masks/challenge-2019-{tvt}-segmentation-masks.csv'),
        'images_dir': os.path.join(oid_root, tvt),
        'masks_dir' : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/all-masks'),
    }


klass_df  = pd.read_csv(get_paths(OID_DIR, 'train')['klass_csv'], names=['MID', 'name'])\
            .rename_axis(index='No').reset_index()
MID_TO_NO = klass_df.set_index('MID')['No'].to_dict()


def get_descs(mask_csv_path):
    mask_pkl_path = os.path.splitext(mask_csv_path)[0] + '.pkl'
    # cache dataset_descs to speed-up.
    if os.path.exists(mask_pkl_path):
        with open(mask_pkl_path, 'rb') as f:
            dataset_descs = pickle.load(f)
    else:
        dataset_descs = create_descs_from_csv(mask_csv_path)
        # cache dataset_descs for next usage.
        with open(mask_pkl_path, 'wb') as f:
            pickle.dump(dataset_descs, f)

    return dataset_descs


def create_descs_from_csv(mask_csv_path):
    # columns: ['ImageID', 'MaskPath', 'LabelName', 'BoxID', 'BoxXMin', 'BoxXMax',
    #           'BoxYMin', 'BoxYMax', 'PredictedIoU', 'Clicks']
    mask_df = pd.read_csv(mask_csv_path, usecols=['ImageID', 'MaskPath', 'LabelName',
                                                  'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax'])
    grouped = mask_df.groupby('ImageID')
    # Detectron2 Standard Dataset Dicts:
    # file_name:      full path to image file;
    # height, width:  shape of image;
    # image_id:
    # annotations:    list[dict]: Each dict contains the following keys,
    #                 of which bbox,bbox_mode and category_id are required:
    #     bbox:
    #     bbox_mode: structure.BoxMode: BoxMode.XYXY_ABS, BoxMode.XYHW_ABS
    #     category_id: integer in [0, n_class -1]
    #     segmentation: polygon: list[list[float]] | RLE: dict
    #     keypoints: ...
    #     iscrowd: ...
    # sem_seg_file_name: used for semantic segmentation.
    dataset_descs = []
    for image_id, group_df in tqdm(grouped):
        annotations = []
        for _, row in group_df.iterrows():
            # 'MaskPath' here actually is mask file name(e.g. 88e582a7b14e34a8_m039xj__6133896f.png)
            mask_fname, _, mid, x1, x2, y1, y2 = \
                row[['MaskPath', 'ImageID', 'LabelName', 'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax']]
            mask_id = os.path.splitext(mask_fname)[0]
            annotations.append({
                'bbox'       : [x1, y1, x2, y2],
                'bbox_mode'  : BoxMode.XYXY_REL,
                'mask_id'    : mask_id,
                'category_id': MID_TO_NO[mid],
            })
        dataset_descs.append({
            'image_id'   : image_id,
            'annotations': annotations
        })

    return dataset_descs


def make_mapper(dataset_name, augmentations: T.AugmentationList = None):
    metadata   = MetadataCatalog.get(dataset_name)
    images_dir = metadata.images_dir
    masks_dir  = metadata.masks_dir

    def _desc_to_example(desc: Dict):
        # Detectron2 Model Input Format:
        # image: Tensor[C, H, W];
        # height, width: output height and width;
        # instances: Instances Object to training, with the following fields:
        #     "gt_boxes":
        #     "gt_classes":
        #     "gt_masks": a PolygonMasks or BitMasks object storing N masks, one for each instance.
        desc       = copy.deepcopy(desc)  # it will be modified by code below
        image_path = os.path.join(images_dir, f'{desc["image_id"]}.jpg')
        # shape: [H, W, C]
        origin_image = detection_utils.read_image(image_path, format="BGR")
        oh, ow, oc = origin_height, origin_width, origin_channels = origin_image.shape
        if augmentations is not None:
            aug_input   = T.AugInput(origin_image)
            transforms  = augmentations(aug_input)
            auged_image = aug_input.image
        else:
            auged_image = origin_image
        ah, aw, ac = auged_height, auged_width, auged_channels = auged_image.shape

        target = Instances(image_size=(ah, aw))
        if 'fill gt_boxes':
            # shape: n_box, 4
            boxes     = np.array([anno['bbox'] for anno in desc['annotations']])
            boxes_abs = boxes * np.array([ow, oh, ow, oh])
            if augmentations is not None:
                # clip transformed bbox to image size
                boxes_auged = transforms.apply_box(np.array(boxes_abs)).clip(min=0)
                boxes_auged = np.minimum(boxes_auged, np.array([aw, ah, aw, ah])[np.newaxis, :])
            else:
                boxes_auged = boxes_abs
            target.gt_boxes = Boxes(boxes_auged)
        if 'fill gt_classes':
            classes = [anno['category_id'] for anno in desc['annotations']]
            classes = torch.tensor(classes, dtype=torch.int64)
            target.gt_classes = classes
        if 'fill gt_masks':
            mask_paths = [os.path.join(masks_dir, f'{anno["mask_id"]}.png') for anno in desc['annotations']]
            masks = np.array(list(map(
                lambda p: cv2.resize(cv2.imread(p, flags=cv2.IMREAD_GRAYSCALE), dsize=(ow, oh)), mask_paths
            )))
            if augmentations is not None:
                masks_auged = np.array(list(map(
                    lambda x: transforms.apply_segmentation(x), masks
                )))
            else:
                masks_auged = masks
            masks_auged = masks_auged > MASK_THRESHOLD
            masks_auged = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks_auged])
            )
            target.gt_masks = masks_auged

        return {
            # expected shape: [C, H, W]
            "image"    : torch.as_tensor(np.ascontiguousarray(auged_image.transpose(2, 0, 1))),
            "height"   : auged_height,
            "width"    : auged_width,
            "instances": target,  # refer: annotations_to_instances()
        }

    return _desc_to_example


# register dataset
for tv in ["train", "validation"]:
    paths = get_paths(OID_DIR, tv)
    ds_name = "oid_" + tv
    if ds_name in DatasetCatalog.list():
        DatasetCatalog.remove(ds_name)
    # register oid dataset dicts.
    DatasetCatalog.register("oid_" + tv, lambda x=paths['mask_csv']: get_descs(x))
    # set oid metadata.
    MetadataCatalog.get("oid_" + tv).set(images_dir=paths['images_dir'])
    MetadataCatalog.get("oid_" + tv).set(masks_dir=paths['masks_dir'])


if __name__ == "__main__":
    if '--config-file' in sys.argv:
        CLI_ARGS = sys.argv[1:]
    else:
        CLI_ARGS = [
            '--config-file', DETECTRON2_DIR + '/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
            '--num-gpus', '1', 'SOLVER.IMS_PER_BATCH', '8', 'SOLVER.BASE_LR', '0.0025',
            'DATASETS.TRAIN', '("coco_2017_val", )',
            # '--opts',
            # 'MODEL.WEIGHTS', './weights/model_final_f10217.pkl',
            # 'MODEL.DEVICE', 'cpu'
        ]

    ARGS = default_argument_parser().parse_args(CLI_ARGS)

    if 'setup(args)':
        args = ARGS
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(
            cfg, args
        )  # if you don't like any of the default setup, write your own setup code

    if 'build_detection_train_loader':
        if 'get_detection_dataset_dicts':
            descs_train: List[Dict] = DatasetCatalog.get("oid_train")
            descs_valid: List[Dict] = DatasetCatalog.get("oid_validation")
        dataset = DatasetFromList(descs_train, copy=False)
        if 'DatasetMapper':
            augs = build_augmentation(cfg, is_train=True)
            mapper = make_mapper('oid_train', T.AugmentationList(augs))
        dataset = MapDataset(dataset, mapper)

        sampler = TrainingSampler(len(dataset))
        data_loader = build_batch_data_loader(
            dataset,
            sampler,
            cfg.SOLVER.IMS_PER_BATCH,
            aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )

    if 'visualization':
        from matplotlib import pyplot as plt
        from detectron2.utils.visualizer import *

        def show(visualizer):
            plt.imshow(visualizer.get_output().get_image())
            plt.show()

        itr = data_loader.__iter__()

        batch = itr.__next__()
        exam = batch[0]
        exam['instances']

        v = Visualizer(exam['image'].numpy().transpose(1, 2, 0))
        show(v)

        def draw_masks(vis: Visualizer, ins: Instances):
            for mask in ins.get_fields()['gt_masks']:
                vis.draw_binary_mask(mask.numpy())

        draw_masks(v, exam['instances'])
        show(v)

        def draw_boxes(vis: Visualizer, ins: Instances):
            for box in ins.get_fields()['gt_boxes']:
                vis.draw_box(box)

        draw_boxes(v, exam['instances'])
        show(v)

        def draw_instances(vis: Visualizer, ins: Instances):
            f = ins.get_fields()
            vis.overlay_instances(boxes=f['gt_boxes'],
                                  labels=f['gt_classes'].tolist(),
                                  masks=f['gt_masks'])

        draw_instances(v, exam['instances'])
        show(v)



