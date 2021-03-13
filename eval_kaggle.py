#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"


import glob
import json
import logging
import os
import platform
import sys

import pandas as pd
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetFromList, MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.data.detection_utils import read_image
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Instances, Boxes

from oid_common_utils import encode_binary_mask
from tqdm import tqdm

# ____________________________ Build Environment _______________________________
N_GPUS  = torch.cuda.device_count()

RUN_ON = 'local' if platform.node() == 'frank-note' \
    else 'google' if platform.node() == 'tyu-det' \
    else 'kaggle'

if RUN_ON == 'kaggle':
    DS_DIR         = '/kaggle/input/open-images-instance-segmentation-rvc-2020/'
    CKPT_DIR       = '/kaggle/input/detectron2/'
    CONFIG_FILE    = os.path.join(CKPT_DIR, 'config.yaml')
    CLASS_CSV_PATH = '../input/oidannotations/open-images-dataset/annotation-instance-segmentation/metadata/challenge-2019-classes-description-segmentable.csv'
    HIER_JSON_PATH = '../input/oidannotations/open-images-dataset/annotation-instance-segmentation/metadata/challenge-2019-label300-segmentable-hierarchy.json'
    BATCH_SIZE     = 8
else:  # local
    DS_DIR         = './datasets/kaggle-oid/'
    CKPT_DIR       = './output-save/'
    CONFIG_FILE    = os.path.join(CKPT_DIR, 'config.yaml')
    CLASS_CSV_PATH = './output-save/class.csv'
    HIER_JSON_PATH = './dataset-oid/annotation-instance-segmentation/metadata/challenge-2019-label300-segmentable-hierarchy.json'
    BATCH_SIZE     = 2

with open(os.path.join(CKPT_DIR, 'last_checkpoint')) as fid:
    ckpt_filename = fid.readline()
    # ckpt_filename = 'model_0044999.pth'
WEIGHTS_PATH = os.path.join(CKPT_DIR, ckpt_filename)

# ____________________________ Config __________________________________________
logger = logging.getLogger("detectron2")

klass_df    = pd.read_csv(CLASS_CSV_PATH, names=['MID', 'name']) \
    .rename_axis(index='No').reset_index()
MID_TO_NO   = klass_df.set_index('MID')['No'].to_dict()
NO_TO_MID   = klass_df.set_index('No')['MID'].to_dict()
KLASS_NAMES = klass_df['name'].to_list()


# ____________________________ Main for Train __________________________________
def get_expanding_dict(hier_json):

    def _expand(entry, parents: tuple):
        ret = [(entry['LabelName'], parents)]
        if 'Subcategory' not in entry.keys():
            return ret
        else:
            parents = (entry['LabelName'], *parents)
            for sub in entry['Subcategory']:
                ret.extend(_expand(sub, parents))
            return ret

    expanding_list = []
    # do not expand top level.
    for ent in hier_json['Subcategory']:
        expanding_list.extend(_expand(ent, ()))

    expanding_dict = {}
    # There is duplicated items in Subcategory, we expand all of their parents.
    for k, v in expanding_list:
        if k in expanding_dict.keys():
            expanding_dict[k] = (*expanding_dict[k], *v)
        else:
            expanding_dict[k] = v

    return expanding_dict


def kaggle_mapper(image_path):
    image_array  = read_image(image_path, format="BGR")
    height, width, channels = image_array.shape
    return {
        'image_id': os.path.basename(image_path).split('.')[0],
        # here: image shape should be [C, H, W], and BGR format
        'image'   : torch.as_tensor(image_array.astype("float32").transpose(2, 0, 1)),
        'height'  : height,
        'width'   : width,
    }


def ptd2_preds_to_kaggle_eval_preds(preds, image_ids, no_to_mid=NO_TO_MID, expanding_dict=None):
    """
    - Model(ptd2 maskrcnn) Output Format
        @see ptd2_preds_to_tfod_eval_preds
    - kaggle evaluation prediction csv format
        | ImageID | ImageWidth | ImageHeight | PredictionString |
        where PredictionString is: LabelA1 ConfidenceA1 EncodedMaskA1 LabelA2 ConfidenceA2 EncodedMaskA2 ...
        注：ImageID is unique key.
    @param preds:
    @param image_ids:
    @param no_to_mid:
    @return:
    """
    def _ins_to_string(klass, score, mask):
        """Convert one instance to prediction string."""
        mid          = no_to_mid[int(klass)]
        confidence   = float(score)
        encoded_mask = encode_binary_mask(mask.numpy()).decode()
        return f"{mid} {confidence} {encoded_mask}"

    pred_dicts = []
    instances_list = [p['instances'].to('cpu') for p in preds]
    for image_id, ins in zip(image_ids, instances_list):  # foreach image instances:
        origin_image_height, origin_image_width = ins.image_size

        if expanding_dict is not None:
            exp_ins_list = []
            for b, s, c, m in zip(ins.pred_boxes, ins.scores, ins.pred_classes, ins.pred_masks):
                child_mid = no_to_mid[int(c)]
                parents = list(map(lambda parent_mid: Instances(ins.image_size,
                                                                pred_boxes=Boxes(b.reshape(-1, 4)),
                                                                scores=torch.as_tensor([s]),
                                                                pred_classes=torch.as_tensor([MID_TO_NO[parent_mid]]),
                                                                pred_masks=m.reshape(-1, *m.shape)),
                                   expanding_dict[child_mid]))
                exp_ins_list.extend(parents)
            ins = Instances.cat([ins, *exp_ins_list])

        prediction_string = " ".join(
            list(map(_ins_to_string, ins.pred_classes, ins.scores, ins.pred_masks))
        )

        pred_dicts.append({
            'ImageID'         : image_id,
            'ImageWidth'      : origin_image_width,
            'ImageHeight'     : origin_image_height,
            'PredictionString': prediction_string,
        })

    return pd.DataFrame.from_records(pred_dicts)


if __name__ == "__main__":
    if '--config-file' in sys.argv:
        CLI_ARGS = sys.argv[1:]
    else:
        CLI_ARGS = [
            '--config-file', f'{CONFIG_FILE}', '--num-gpus', f'{N_GPUS}',
            '--eval-only',
            '--resume',
            'MODEL.WEIGHTS', f'{WEIGHTS_PATH}',
            'TEST.DETECTIONS_PER_IMAGE', '200',
            # 'MODEL.ROI_HEADS.NMS_THRESH_TEST', '0.7',
        ]
        if RUN_ON == 'local':
            CLI_ARGS.extend(['MODEL.DEVICE', 'cpu'])
    args = default_argument_parser().parse_args(CLI_ARGS)

    if 'setup(args)':
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)  # if you don't like any of the default setup, write your own setup code

    if 'build dataloader':
        file_paths = glob.glob(DS_DIR + 'test/*.jpg')
        # import numpy as np
        # file_paths = np.random.choice(file_paths, size=200, replace=False)
        ds = DatasetFromList(file_paths)
        ds = MapDataset(ds, map_func=kaggle_mapper)

        sampler = InferenceSampler(len(ds))
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, BATCH_SIZE, drop_last=False)

        data_loader = torch.utils.data.DataLoader(ds,
                                                  batch_sampler=batch_sampler,
                                                  collate_fn = trivial_batch_collator,)

    if 'create model and load weights':
        meta_arch = cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.eval()
        model.to(torch.device(cfg.MODEL.DEVICE))

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

    if 'expanding parents category':
        with open(HIER_JSON_PATH, 'r') as fid:
            hier = json.load(fid)
        exp_dict = get_expanding_dict(hier)

    all_kaggle_preds = []
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        for bat_inp in tqdm(data_loader):
            bat_ids   = [x['image_id'] for x in bat_inp]
            bat_preds = model(bat_inp)

            bat_kaggle_preds = ptd2_preds_to_kaggle_eval_preds(bat_preds, bat_ids, no_to_mid=NO_TO_MID, expanding_dict=exp_dict)
            all_kaggle_preds.append(bat_kaggle_preds)

    submission = pd.concat(all_kaggle_preds)
    submission.to_csv('submission.csv', index=False)




