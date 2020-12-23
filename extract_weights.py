#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"

import os
import pickle
import sys
import platform
from collections import OrderedDict
from copy import deepcopy

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import META_ARCH_REGISTRY

# ____________________________ Build Environment _______________________________
from fvcore.common.file_io import PathManager

RUN_ON = 'local' if platform.node() == 'frank-note' else 'google-cloud'

# ____________________________ Config __________________________________________
N_GPU  = torch.cuda.device_count()
DS_TYPE = 'oid'  # oid or coco
if RUN_ON == 'local':
    DETECTRON2_DIR = '/data/venv-pytorch/detectron2'
else:
    DETECTRON2_DIR = '/home/jupyter/detectron2'
CONFIG_FILE = os.path.join(DETECTRON2_DIR,
                           'configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')


if __name__ == "__main__":
    if '--config-file' in sys.argv:
        CLI_ARGS = sys.argv[1:]
    else:
        CLI_ARGS = [
            '--config-file', f'{CONFIG_FILE}', '--num-gpus', f'{N_GPU}',
            'MODEL.WEIGHTS', './weights/model_final_f10217.pkl',
            'MODEL.DEVICE', 'cpu'
        ]
    ARGS = default_argument_parser().parse_args(CLI_ARGS)

    args = ARGS
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

    # if 'build_model(cfg)':
    #     meta_arch = cfg.MODEL.META_ARCHITECTURE
    #     model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    #     model.to(torch.device(cfg.MODEL.DEVICE))
    #
    # if 'load_pretrained_weights':
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=False
    #     )

    if 'extract_weights_without_roi_heads':
        with open(cfg.MODEL.WEIGHTS, 'rb') as f:
            ckpt = pickle.load(f, encoding='latin1')

        weight_names = deepcopy(list(ckpt["model"].keys()))
        for k in weight_names:
            if k.startswith('roi_heads'):
                ckpt["model"].pop(k)

        save_path = cfg.MODEL.WEIGHTS.replace('.pkl', '_without_roi_heads.pkl')
        with PathManager.open(save_path, "wb") as f:
            pickle.dump(ckpt, f)




