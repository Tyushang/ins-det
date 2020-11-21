#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"


import copy
import datetime
import logging
import os
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

from prepare_oid import make_mapper

# ____________________________ Config __________________________________________
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
CONFIG = None


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


# __________________ For Debug _____________________________
tic = None

mem_stats_df = MemStatsInDataFrame()
# tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# tcp_sock.connect(('localhost', 9999))


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

    if N_GPU > 0:
        # __________________ For Debug _____________________________
        # mem_stats_df.record('Before-Build-Model')
        if 'build_model(cfg)':
            meta_arch = cfg.MODEL.META_ARCHITECTURE
            model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
            model.to(torch.device(cfg.MODEL.DEVICE))
        # __________________ For Debug _____________________________
        # mem_stats_df.record('After-Build-Model')

    if 'do-train':
        if 'build_detection_train_loader':
            if 'coco_2017_train' in cfg.DATASETS.TRAIN:
                descs_train: List[Dict] = DatasetCatalog.get("coco_2017_train")
                dataset = DatasetFromList(descs_train, copy=False)
                mapper = DatasetMapper(cfg, True)
            else:  # Open-Image-Dataset
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

            # __________________ For Debug _____________________________
            # mem_stats_df.record('Before-Iteration')
            with EventStorage(start_iter) as storage:
                for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                    iteration = iteration + 1
                    storage.step()
                    # __________________ For Debug _____________________________
                    # bat_image_shape = [x['image'].shape for x in data]
                    # bat_n_anno = [len(x['instances']) for x in data]
                    # print('_'*10 + f'Image Shape: {bat_image_shape}; Num Anno: {bat_n_anno}')

                    # __________________ For Debug _____________________________
                    # mem_stats_df.record('Before-Forward')
                    loss_dict = model(data)
                    # __________________ For Debug _____________________________
                    # mem_stats_df.record('After-Forward')

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
                        # __________________ For Debug _____________________________
                        # mem_summary = torch.cuda.memory_summary()
                        # tcp_sock.send(mem_summary.encode('utf-8'))
                        global tic
                        if tic is None:
                            tic = datetime.datetime.now()
                        else:
                            toc = datetime.datetime.now()
                            print('_' * 35 + f'Time Elapsed: {(toc - tic).total_seconds()} s')
                            tic = toc

                    periodic_checkpointer.step(iteration)

            # __________________ For Debug _____________________________
            # mem_stats_df.dump('./mem_stats_df.csv')


if __name__ == "__main__":
    if '--config-file' in sys.argv:
        CLI_ARGS = sys.argv[1:]
    else:
        CLI_ARGS = [
            '--config-file', 'configs-oid/mask_rcnn_R_50_FPN_3x.yaml',
            '--num-gpus', f'{N_GPU}',
            'SOLVER.IMS_PER_BATCH', '4', 'SOLVER.BASE_LR', '0.0025',
            'DATASETS.TRAIN', '("oid_train", )',
            'DATASETS.TEST', '("oid_validation", )',
            # For Debug ____________________
            # 'SOLVER.MAX_ITER', '20_000',
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

