#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"


import copy
import logging
import os
import sys
import pickle
import platform
from typing import *

import numpy as np
import pandas as pd
from cv2 import cv2

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetFromList, MapDataset, build_batch_data_loader, \
    DatasetMapper
from detectron2.data.detection_utils import build_augmentation
from detectron2.data.samplers import TrainingSampler
from detectron2.engine import default_argument_parser, default_setup, PeriodicCheckpointer, launch
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.structures import BoxMode, Instances, Boxes, BitMasks
from detectron2.data import detection_utils
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage
from detectron2.data import transforms as T
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

N_GPU  = torch.cuda.device_count()
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


def make_mapper(dataset_name, augmentations=None):
    metadata   = MetadataCatalog.get(dataset_name)
    images_dir = metadata.images_dir
    masks_dir  = metadata.masks_dir
    if augmentations is not None:
        augmentations = T.AugmentationList(augmentations)

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
            boxes_abs = boxes * np.array([aw, ah, aw, ah])
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
            "image"    : torch.as_tensor(np.ascontiguousarray(origin_image.transpose(2, 0, 1))),
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


# def do_test(cfg, model):
#     results = OrderedDict()
#     for dataset_name in cfg.DATASETS.TEST:
#         data_loader = build_detection_test_loader(cfg, dataset_name)
#         evaluator = get_evaluator(
#             cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
#         )
#         results_i = inference_on_dataset(model, data_loader, evaluator)
#         results[dataset_name] = results_i
#         if comm.is_main_process():
#             logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#             print_csv_format(results_i)
#     if len(results) == 1:
#         results = list(results.values())[0]
#     return results

DATA_LOADER = None


def main(args):
    print('_' * 60 + f'\nmain <- {args}')
    if 'setup(args)':
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(
            cfg, args
        )  # if you don't like any of the default setup, write your own setup code

    if N_GPU > 0:
        if 'build_model(cfg)':
            meta_arch = cfg.MODEL.META_ARCHITECTURE
            model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
            model.to(torch.device(cfg.MODEL.DEVICE))

    if 'do-train':
        if 'build_detection_train_loader':
            # oid dataset.
            if 'get_detection_dataset_dicts':
                descs_train: List[Dict] = DatasetCatalog.get("oid_train")
                descs_valid: List[Dict] = DatasetCatalog.get("oid_validation")
            dataset = DatasetFromList(descs_train, copy=False)
            if 'DatasetMapper':
                augs = build_augmentation(cfg, is_train=True)
            dataset = MapDataset(dataset, make_mapper('oid_train', augs))
            # coco dataset.
            # descs_train: List[Dict] = DatasetCatalog.get("coco_2017_train")
            # dataset = DatasetFromList(descs_train, copy=False)
            # mapper = DatasetMapper(cfg, True)
            # dataset = MapDataset(dataset, mapper)

            sampler = TrainingSampler(len(dataset))
            data_loader = build_batch_data_loader(
                dataset,
                sampler,
                cfg.SOLVER.IMS_PER_BATCH,
                aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
            )
            global DATA_LOADER
            DATA_LOADER = data_loader

        if N_GPU > 0:
            cfg, model, resume = cfg, model, False

            model.train()
            optimizer = build_optimizer(cfg, model)
            scheduler = build_lr_scheduler(cfg, optimizer)

            checkpointer = DetectionCheckpointer(
                model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
            )
            start_iter = (
                    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
            )
            max_iter = cfg.SOLVER.MAX_ITER

            periodic_checkpointer = PeriodicCheckpointer(
                checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
            )

            writers = (
                [
                    CommonMetricPrinter(max_iter),
                    JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
                    TensorboardXWriter(cfg.OUTPUT_DIR),
                ]
                if comm.is_main_process()
                else []
            )
            logger = logging.getLogger("detectron2")
            logger.info("Starting training from iteration {}".format(start_iter))
            with EventStorage(start_iter) as storage:
                for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                    iteration = iteration + 1
                    storage.step()

                    loss_dict = model(data)
                    losses = sum(loss_dict.values())
                    assert torch.isfinite(losses).all(), loss_dict

                    loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    if comm.is_main_process():
                        storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                    scheduler.step()

                    # if (
                    #     cfg.TEST.EVAL_PERIOD > 0
                    #     and iteration % cfg.TEST.EVAL_PERIOD == 0
                    #     and iteration != max_iter
                    # ):
                    #     do_test(cfg, model)
                    #     # Compared to "train_net.py", the test results are not dumped to EventStorage
                    #     comm.synchronize()

                    if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                        for writer in writers:
                            writer.write()
                    periodic_checkpointer.step(iteration)


if __name__ == "__main__":
    if '--config-file' in sys.argv:
        CLI_ARGS = sys.argv[1:]
    else:
        CLI_ARGS = [
            '--config-file', 'configs-oid/mask_rcnn_R_50_FPN_3x.yaml',
            '--num-gpus', f'{N_GPU}',
            'SOLVER.IMS_PER_BATCH', '2', 'SOLVER.BASE_LR', '0.0025',
            'DATASETS.TRAIN', '("oid_train", )',
            'DATASETS.TEST', '("oid_validation", )',
            # INPUT.FORMAT?  INPUT.MASK_FORMAT?
            # '--opts',
            # 'MODEL.WEIGHTS', './weights/model_final_f10217.pkl',
            # 'MODEL.DEVICE', 'cpu'
        ]
    ARGS = default_argument_parser().parse_args(CLI_ARGS)

    print("Command Line Args:", ARGS)

    # if 'dataset-statistic':
    #     import pandas as pd
    #     oid_train_dicts: List[Dict] = DatasetCatalog.get("oid_train")
    #     oid_train_stat = pd.DataFrame(list(map(
    #         lambda x: len(x['annotations']), oid_train_dicts
    #     )))
    #     oid_train_dicts: List[Dict] = DatasetCatalog.get("coco_2017_train")
    #     coco_train_stat = pd.DataFrame(list(map(
    #         lambda x: len(x['annotations']), oid_train_dicts
    #     )))

    if N_GPU == 0:
        main(ARGS)
    else:
        launch(
            main,
            ARGS.num_gpus,
            num_machines=ARGS.num_machines,
            machine_rank=ARGS.machine_rank,
            dist_url=ARGS.dist_url,
            args=(ARGS,),
        )

