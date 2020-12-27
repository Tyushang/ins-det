#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"
import contextlib
import copy
import datetime
import io
import itertools
import json
import logging
import os
import random
import socket
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
    DatasetMapper, build_detection_test_loader
from detectron2.data.build import trivial_batch_collator
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.data.detection_utils import build_augmentation
from detectron2.data.samplers import TrainingSampler, InferenceSampler
from detectron2.engine import default_argument_parser, default_setup, PeriodicCheckpointer, launch
from detectron2.evaluation import inference_on_dataset, print_csv_format, SemSegEvaluator, COCOEvaluator, \
    COCOPanopticEvaluator, CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, PascalVOCDetectionEvaluator, \
    LVISEvaluator, DatasetEvaluators, DatasetEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json, _evaluate_predictions_on_coco
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.structures import BoxMode, Instances, Boxes, BitMasks
from detectron2.data import detection_utils
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage
from detectron2.data import transforms as T
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tqdm import tqdm

from eval_utils import get_evaluator2
from prepare_oid import make_mapper

# ____________________________ Build Environment _______________________________
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# clear_descs_cache()

# ____________________________ Config __________________________________________
N_GPU  = torch.cuda.device_count()

DATA_LOADER = None
CONFIG      = None
TIC         = None

logger = logging.getLogger("detectron2")


EVALUATOR = None


# ____________________________ Main for Train __________________________________
def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        # data_loader = build_detection_test_loader(cfg, dataset_name)
        if 'build_detection_test_loader':
            if 'get_detection_dataset_dicts':
                descs_valid: List[Dict] = DatasetCatalog.get(dataset_name)
            # validation dataset is too large.
            random.seed(2020)
            descs_valid = random.sample(descs_valid, k=10)
            dataset = DatasetFromList(descs_valid)
            if 'DatasetMapper':
                mapper = make_mapper(dataset_name, is_train=False, augmentations=None)
            dataset = MapDataset(dataset, mapper)

            sampler = InferenceSampler(len(dataset))
            # Always use 1 image per worker during inference since this is the
            # standard when reporting inference time in papers.
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

            data_loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                batch_sampler=batch_sampler,
                collate_fn=trivial_batch_collator,
            )

        evaluator = get_evaluator2(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        global EVALUATOR
        EVALUATOR = evaluator

        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        # TODO: Multiprocessing?
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            if 'print_csv_format(results_i)':
                for tsk, res in results_i.items():
                    global RES
                    RES = res
                    res_df = pd.DataFrame(pd.Series(res, name='value'))
                    res_df = res_df[res_df['value'].notna()]
                    # res_df = res_df[res_df['value'] > 0]
                    res_df.index = res_df.index.map(lambda x: '/'.join(x.split('/')[1:]))
                    pd.set_option('display.max_rows', None)
                    print(res_df)
                    pd.reset_option('display.max_rows')


RES = None


def main(args):
    if 'setup(args)':
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(
            cfg, args
        )  # if you don't like any of the default setup, write your own setup code
        global CONFIG
        CONFIG = cfg

    # __________________ For Debug _____________________________
    # mem_stats_df.record('Before-Build-Model')
    if 'build_model(cfg)':
        meta_arch = cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
    # __________________ For Debug _____________________________
    # mem_stats_df.record('After-Build-Model')

    if 'evaluation':
        checkpoint = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS_PATH, resume=args.resume
        )
        do_test(cfg, model)


if __name__ == "__main__":
    if '--config-file' in sys.argv:
        CLI_ARGS = sys.argv[1:]
    else:
        CLI_ARGS = [
            # '--config-file', 'configs-oid/mask_rcnn_R_50_FPN_3x.yaml',
            '--config-file', './output-save/config.yaml',
            # 'MODEL.WEIGHTS', './weights/model_final_f10217.pkl',
            'MODEL.WEIGHTS', './output-save/model_0017999.pth',
            'MODEL.DEVICE', 'cpu',
            'SOLVER.IMS_PER_BATCH', '4', 'SOLVER.BASE_LR', '0.0025',
            'DATASETS.TRAIN', '("oid_train", )',
            'DATASETS.TEST', '("oid_validation", )',
            # For Debug ____________________
            # 'SOLVER.MAX_ITER', '20_000',
            # INPUT.FORMAT?  INPUT.MASK_FORMAT?
            # '--opts',
            # 'MODEL.DEVICE', 'cpu'
        ]
    ARGS = default_argument_parser().parse_args(CLI_ARGS)

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

