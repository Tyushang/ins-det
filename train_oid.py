#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"
import contextlib
import copy
import datetime
import gc
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
from detectron2.utils.logger import create_small_table
from fvcore.common.file_io import PathManager
from memory_profiler import profile
from pycocotools.coco import COCO
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from eval_utils import MyCocoEvaluator, get_evaluator2
from prepare_oid import make_mapper

# ____________________________ Build Environment _______________________________
RUN_ON = 'local' if platform.node() == 'frank-note' \
    else 'google' if platform.node() == 'tyu-det' \
    else 'kaggle'

if RUN_ON == 'local':
    DETECTRON2_DIR = '/data/venv-pytorch/detectron2'
else:
    DETECTRON2_DIR = '/home/tyushang_gmail_com/jupyter/detectron2'

# ____________________________ Config __________________________________________
N_GPUS  = torch.cuda.device_count()
N_IMAGES_TRAIN = 847_997

IMS_PER_GPU = 4
BATCH_SIZE  = N_GPUS * IMS_PER_GPU

N_EPOCHS = 1
N_STEPS  = (N_IMAGES_TRAIN * N_EPOCHS) // BATCH_SIZE

N_IMAGES_PER_TEST = 800

MILESTONES_RATIO = (0.77, 0.92)
MILESTONES       = tuple([int(m * N_STEPS) for m in MILESTONES_RATIO])

DS_TYPE = 'oid'  # oid or coco
if DS_TYPE == 'oid':
    CONFIG_FILE = './configs-oid/mask_rcnn_R_50_FPN_3x.yaml'
    WEIGHTS  = './weights/model_final_f10217_without_roi_heads.pkl'
    # WEIGHTS  = './output-save/model_0017999.pth'
    DS_TRAIN = 'oid_train'
    DS_VALID = 'oid_validation'
else:
    CONFIG_FILE = os.path.join(DETECTRON2_DIR, 'configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    WEIGHTS  = './weights/model_final_f10217.pkl'
    DS_TRAIN = 'coco_2017_train'
    DS_VALID = 'coco_2017_val'


# ____________________________ For Debug _______________________________________
class MemStatsInDataFrame:
    def __init__(self):
        self.all_ms_df: pd.DataFrame = None
        self.sn = 0

    def record(self, memo=None):
        ms = torch.cuda.memory_stats()
        index = [f'{self.sn:03d}:{memo or "unnamed"}']
        ms_df = pd.DataFrame([ms.values()], columns=ms.keys(), index=index)
        if self.all_ms_df is None:
            self.all_ms_df = ms_df
        else:
            self.all_ms_df = self.all_ms_df.append(ms_df)
        self.sn += 1

    def dump(self, path):
        self.all_ms_df.to_csv(path, index_label='memo')


mem_stats_df = MemStatsInDataFrame()
# tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# tcp_sock.connect(('localhost', 9999))

DATA_LOADER = None
CONFIG      = None
TIC         = None

logger = logging.getLogger("detectron2")


# ____________________________ Main for Train __________________________________
# noinspection DuplicatedCode,PyMethodMayBeStatic

@profile
def do_test(cfg, model):
    for dataset_name in cfg.DATASETS.TEST:
        # data_loader = build_detection_test_loader(cfg, dataset_name)
        if 'build_detection_test_loader':
            if dataset_name == 'coco_2017_val':
                dicts_valid: List[Dict] = DatasetCatalog.get(dataset_name)
                if "filter_empty and has_instances":
                    ...
                ds_valid = DatasetFromList(dicts_valid, copy=False)
                mapper = DatasetMapper(cfg, is_train=False)
            else:  # Open-Image-Dataset
                if 'get_detection_dataset_dicts':
                    descs_get: List[Dict] = DatasetCatalog.get(dataset_name)
                # validation dataset is too large.
                random.seed(2020)
                descs_valid = random.choices(descs_get, k=N_IMAGES_PER_TEST)
                # TODO: clear cache.
                ds_valid = DatasetFromList(descs_valid)
                if 'DatasetMapper':
                    mapper = make_mapper(dataset_name, is_train=False, augmentations=None)

            ds_valid = MapDataset(ds_valid, mapper)

            sampler = InferenceSampler(len(ds_valid))
            # Always use 1 image per worker during inference since this is the
            # standard when reporting inference time in papers.
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

            data_loader = torch.utils.data.DataLoader(
                ds_valid,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                batch_sampler=batch_sampler,
                collate_fn=trivial_batch_collator,
            )

        evaluator = get_evaluator2(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )

        results_i = inference_on_dataset(model, data_loader, evaluator)
        # torch.cuda.empty_cache()
        # if comm.is_main_process():
        #     comm.synchronize()
        #     state_and_ids_list = comm.gather(evaluator.state_and_ids_list, dst=0)
        #
        #     from object_detection.utils import object_detection_evaluation as tfod_evaluation
        #     merged = tfod_evaluation.OpenImagesChallengeEvaluator(evaluator, True)
        #     for state, ids in state_and_ids_list:
        #         merged.merge_internal_state(ids, state)
        #
        #     metrics = merged.evaluate()
        #     results_i = OrderedDict({'instance-segmentation': metrics})
        #
        #     logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        #     # print_csv_format(results_i)
        #     for tsk, res in results_i.items():
        #         res_df = pd.DataFrame(pd.Series(res, name='value'))
        #         res_df = res_df[res_df['value'].notna()]
        #         # res_df = res_df[res_df['value'] > 0]
        #         res_df.index = res_df.index.map(lambda x: '/'.join(x.split('/')[1:]))
        #         pd.set_option('display.max_rows', None)
        #         print(res_df)
        #         pd.reset_option('display.max_rows')


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
        global CONFIG
        CONFIG = cfg

    if True:  # N_GPU > 0:
        # __________________ For Debug _____________________________
        # mem_stats_df.record('Before-Build-Model')
        if 'build_model(cfg)':
            meta_arch = cfg.MODEL.META_ARCHITECTURE
            model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
            model.to(torch.device(cfg.MODEL.DEVICE))
        # __________________ For Debug _____________________________
        # mem_stats_df.record('After-Build-Model')

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    if 'do-train':
        if 'build_detection_train_loader':
            if 'coco_2017_train' in cfg.DATASETS.TRAIN:
                descs_train: List[Dict] = DatasetCatalog.get("coco_2017_train")
                ds_train = DatasetFromList(descs_train, copy=False)
                mapper = DatasetMapper(cfg, True)
            else:  # Open-Image-Dataset
                if 'get_detection_dataset_dicts':
                    descs_train: List[Dict] = DatasetCatalog.get("oid_train")
                ds_train = DatasetFromList(descs_train, copy=False)
                if 'DatasetMapper':
                    augs = build_augmentation(cfg, is_train=True)
                    mapper = make_mapper('oid_train', is_train=True, augmentations=T.AugmentationList(augs))
            ds_train = MapDataset(ds_train, mapper)

            sampler = TrainingSampler(len(ds_train))
            data_loader = build_batch_data_loader(
                ds_train,
                sampler,
                cfg.SOLVER.IMS_PER_BATCH,
                aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
            )
            global DATA_LOADER
            DATA_LOADER = data_loader

        if N_GPUS > 0:
            cfg, model, resume = cfg, model, args.resume

            model.train()
            optimizer = build_optimizer(cfg, model)
            scheduler = build_lr_scheduler(cfg, optimizer)

            checkpointer = DetectionCheckpointer(
                model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
            )
            # "iteration" always be loaded whether resume or not.
            # "model" state_dict will always be loaded whether resume or not.
            start_iter = (
                    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
            )
            max_iter = cfg.SOLVER.MAX_ITER
            # optimizer and scheduler will be resume to checkpointer.checkpointables[*] if resume is True
            if resume:
                optimizer = checkpointer.checkpointables['optimizer']
                scheduler = checkpointer.checkpointables['scheduler']

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

                    # __________________ Checkpoint / Test / Metrics ___________
                    periodic_checkpointer.step(iteration)

                    if (
                        cfg.TEST.EVAL_PERIOD > 0
                        and iteration % cfg.TEST.EVAL_PERIOD == 0
                        and iteration != max_iter
                    ):
                        do_test(cfg, model)
                        # Compared to "train_net.py", the test results are not dumped to EventStorage
                        comm.synchronize()

                    if iteration - start_iter > 5 and (iteration % 100 == 0 or iteration == max_iter):
                        for writer in writers:
                            writer.write()
                        # __________________ For Debug _____________________________
                        # mem_summary = torch.cuda.memory_summary()
                        # tcp_sock.send(mem_summary.encode('utf-8'))
                        global TIC
                        if TIC is None:
                            TIC = datetime.datetime.now()
                        else:
                            toc = datetime.datetime.now()
                            logger.info('_' * 35 + f'Time Elapsed: {(toc - TIC).total_seconds()} s')
                            TIC = toc

            # __________________ For Debug _____________________________
            # mem_stats_df.dump('./mem_stats_df.csv')


if __name__ == "__main__":
    if '--config-file' in sys.argv:
        CLI_ARGS = sys.argv[1:]
    else:
        CLI_ARGS = [
            '--config-file', f'{CONFIG_FILE}', '--num-gpus', f'{N_GPUS}',
            '--dist-url', 'auto',
            # '--eval-only',
            '--resume',
            'MODEL.WEIGHTS', f'{WEIGHTS}',
            'DATASETS.TRAIN', f'("{DS_TRAIN}", )', 'DATASETS.TEST', f'("{DS_VALID}", )',
            'SOLVER.IMS_PER_BATCH', f'{BATCH_SIZE}',
            'SOLVER.BASE_LR', '0.0025',
            'SOLVER.MAX_ITER', f'{N_STEPS}',
            'SOLVER.STEPS', f'{MILESTONES}',
            'TEST.EVAL_PERIOD', '10',
            'SOLVER.CHECKPOINT_PERIOD', '3000',
            # For Debug ____________________
            # INPUT.FORMAT?  INPUT.MASK_FORMAT?
            # 'MODEL.DEVICE', 'gpu' if N_GPU > 0 else 'cpu',
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

    if N_GPUS == 0:
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

