#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"


import copy
import glob
import os
import pickle
import platform
from functools import partial
from multiprocessing import Pool
from typing import *

import numpy as np
import pandas as pd
import pycocotools
import torch
from cv2 import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode, Instances, Boxes, BitMasks
from tqdm import tqdm

from oid_common_utils import id_from_path

# ____________________________ Environment _____________________________________
RUN_ON = 'local' if platform.node() == 'frank-note' else 'google-cloud'

if RUN_ON == 'local':
    OID_DIR    = '/data/venv-tensorflow2/open-images-dataset'
    CHUNK_SIZE = 0x10
else:
    OID_DIR    = '/home/jupyter/datasets/oid'
    CHUNK_SIZE = 0x100


def get_paths(oid_root, tvt):
    """
    @param oid_root:
    @param tvt: 'train', 'validation' or 'test'
    Dataset Dir Tree:
    ROOT:
    |-- annotation-instance-segmentation
    |   |-- metadata
    |   |   |-- challenge-2019-classes-description-segmentable.csv      # class_csv      for short.
    |   |   |-- challenge-2019-label300-segmentable-hierarchy.json      # hierarchy_json for short.
    |   |   |-- oid_object_detection_challenge_500_label_map.pbtxt      # label_map_pbtxt for short. oid_challenge_evaluation need it.
    |   |-- train
    |   |   |-- cache
    |   |   |   |-- image-shape.csv                                     # cache image height and width.          shape_csv  for short.
    |   |   |   |-- coco-format.json                                    # coco Standard Format .json Cache-File. coco_json  for short.
    |   |   |   |-- joined-mask.csv                                     # mask join image shape and bbox.        joined_csv for short.
    |   |   |   |-- detectron2-dicts.pkl                                # detectron2 Standard Format Cache-File. dicts_pkl  for short.
    |   |   |-- challenge-2019-train-masks
    |   |   |   |-- challenge-2019-train-segmentation-masks.csv         # mask_csv for short.
    |   |   |   |-- challenge-2019-train-masks-[0~f].zip
    |   |   |-- all-masks                                               # N_MASK: 2125530
    |   |   |-- challenge-2019-train-segmentation-bbox.csv              # bbox_csv for short.
    |   |   |-- challenge-2019-train-segmentation-labels.csv            # image_label_csv for short.
    |   |-- validation
    |       |-- cache
    |       |   |-- image-shape.csv                                     # cache image height and width.          shape_csv  for short.
    |       |   |-- coco-format.json                                    # coco Standard Format .json Cache-File. coco_json  for short.
    |       |   |-- joined-mask.csv                                     # mask join image shape and bbox.        joined_csv for short.
    |       |   |-- detectron2-dicts.pkl                                # detectron2 Standard Format Cache-File. dicts_pkl  for short.
    |       |-- challenge-2019-validation-masks
    |       |   |-- challenge-2019-validation-segmentation-masks.csv
    |       |   |-- challenge-2019-validation-masks-[0~f].zip
    |       |-- all-masks                                               # N_MASK: 23366
    |       |-- challenge-2019-validation-segmentation-bbox.csv
    |       |-- challenge-2019-validation-segmentation-labels.csv
    |-- train       # N_IMAGE: 1_743_042 , mask_csv only used 848_512 images(847_997 images with height and width <= 1024)?
    |-- validation  # N_IMAGE:    41_620 , mask_csv only used  12_965 images( 12_950 images with height and width <= 1024)?
    |-- test        # N_IMAGE:   125_436
    """
    from argparse import Namespace
    return Namespace(**{
        'klass_csv'      : os.path.join(oid_root, 'annotation-instance-segmentation/metadata/'
                                        + 'challenge-2019-classes-description-segmentable.csv'),
        'label_map_pbtxt': os.path.join(oid_root, 'annotation-instance-segmentation/metadata/'
                                        + 'oid_object_detection_challenge_500_label_map.pbtxt'),
        'mask_csv'       : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/'
                                        + f'challenge-2019-{tvt}-masks/'
                                        + f'challenge-2019-{tvt}-segmentation-masks.csv'),
        'bbox_csv'       : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/'
                                        + f'challenge-2019-{tvt}-segmentation-bbox.csv'),
        'image_label_csv': os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/'
                                        + f'challenge-2019-{tvt}-segmentation-labels.csv'),
        'images_dir'     : os.path.join(oid_root, tvt),
        'masks_dir'      : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/all-masks'),
        'cache'          : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/cache'),
        'shape_csv'      : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/cache/image_shape.csv'),
        'coco_json'      : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/cache/coco-format.json'),
        'joined_csv'     : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/cache/joined-mask.csv'),
        'descs_pkl'      : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/cache/oid-descs.pkl'),
        'dicts_pkl'      : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/cache/detectron2-dicts.pkl'),
        'tfod_mask_pkl'  : os.path.join(oid_root, f'annotation-instance-segmentation/{tvt}/cache/tfod-mask.pkl'),
    })


if not os.path.exists(get_paths(OID_DIR, 'train').cache):
    os.makedirs(get_paths(OID_DIR, 'train').cache, exist_ok=True)
if not os.path.exists(get_paths(OID_DIR, 'validation').cache):
    os.makedirs(get_paths(OID_DIR, 'validation').cache, exist_ok=True)

klass_df    = pd.read_csv(get_paths(OID_DIR, 'train').klass_csv, names=['MID', 'name'])\
              .rename_axis(index='No').reset_index()
MID_TO_NO   = klass_df.set_index('MID')['No'].to_dict()
NO_TO_MID   = klass_df.set_index('No')['MID'].to_dict()
KLASS_NAMES = klass_df['name'].to_list()


def remove_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File removed: {file_path}")
    else:
        print(f"File not exist: {file_path}")


def clear_shape_csv():
    for tv in ["train", "validation"]:
        paths = get_paths(OID_DIR, tv)
        remove_if_exists(paths.shape_csv)


def clear_pkl():
    for tv in ["train", "validation"]:
        paths = get_paths(OID_DIR, tv)
        remove_if_exists(paths.descs_pkl)
        remove_if_exists(paths.dicts_pkl)


# ____________________________ Configurations __________________________________
MASK_THRESHOLD = 127


# ____________________________ Helper Functions ________________________________
class DescPipeline:

    def __init__(self, funcs: List[Callable], cache_paths: List[str] = None):
        if cache_paths is not None:
            assert len(funcs) == len(cache_paths)
        self.funcs       = funcs
        self.cache_paths = cache_paths

    def __call__(self):
        if self.cache_paths is not None:
            funcs = []
            for path, func in zip(reversed(self.cache_paths), reversed(self.funcs)):
                if path is not None and os.path.exists(path):
                    funcs.insert(0, partial(load_pkl, path=path))
                    break
                else:
                    funcs.insert(0, func)
        else:
            funcs = self.funcs
        # TODO: funcs are responsible for dumping cache.
        inp = out = funcs[0]()
        for func in funcs[1:]:
            inp = out = func(inp)

        return out


def load_pkl(path):
    print(f'Loading {path}')
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret


def get_image_size_by_pil(image_path):
    """@return: image_id, width, height"""
    image_id = id_from_path(image_path)
    from PIL import Image
    im = Image.open(image_path)
    return (image_id, ) + im.size


def create_image_shape_csv(is_train):
    paths = get_paths(OID_DIR, 'train' if is_train else 'validation')
    shape_csv_path = paths.shape_csv
    images_dir     = paths.images_dir

    from datetime import datetime
    tic = datetime.now()
    with Pool() as pool:
        shapes = tqdm(pool.imap(get_image_size_by_pil, glob.glob(f'{images_dir}/*.jpg'), chunksize=CHUNK_SIZE))
        df = pd.DataFrame(shapes, columns=['ImageID', 'Width', 'Height']).set_index('ImageID')
        df.to_csv(shape_csv_path)
    print('_'*30 + f'Time: {(datetime.now() - tic).total_seconds()}')


# ____________________________ Main Functions __________________________________
def get_oid_descs(is_train):
    paths = get_paths(OID_DIR, 'train' if is_train else 'validation')
    # cache oid_descs to speed-up.
    if os.path.exists(paths.descs_pkl):
        with open(paths.descs_pkl, 'rb') as f:
            detectron2_dicts = pickle.load(f)
    else:
        if not os.path.exists(paths.shape_csv):
            create_image_shape_csv(is_train)
        detectron2_dicts = create_descs_from_csv(paths.mask_csv, paths.shape_csv, paths.joined_csv)
        # cache oid_descs for next usage.
        with open(paths.descs_pkl, 'wb') as f:
            pickle.dump(detectron2_dicts, f)

    return detectron2_dicts


def create_descs_from_csv(mask_csv_path, image_shape_csv_path, joined_csv_path=None):
    # columns: ['ImageID', 'MaskPath', 'LabelName',    'BoxID', 'BoxXMin', 'BoxXMax',
    #           'BoxYMin', 'BoxYMax',  'PredictedIoU', 'Clicks']
    mask_df = pd.read_csv(mask_csv_path, usecols=['ImageID', 'MaskPath', 'LabelName',
                                                  'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax'])
    if 'add-height-width':
        # columns: ['ImageID', 'Width', 'Height']
        shape_df = pd.read_csv(image_shape_csv_path)
        joined = mask_df.set_index('ImageID').join(shape_df.set_index('ImageID'))
        if joined_csv_path is not None:
            joined.to_csv(joined_csv_path)

    if 'filter-images-with-height-and-width<=1024':
        joined = joined[(joined['Height'] <= 1024) & (joined['Width'] <= 1024)]

    if 'add-iscrowd':
        # TODO: there's no box id in bbox_csv, can not join.
        # columns: ['ImageID' 'LabelName' 'XMin' 'XMax' 'YMin' 'YMax' 'IsGroupOf']
        # bbox_df  = pd.read_csv(bbox_csv_path).set_index('mask_id')
        ...

    grouped = joined.groupby(by='ImageID')  # ImageID is index.

    detectron2_dicts = []
    with Pool() as pool:
        for x in tqdm(pool.imap(to_oid_desc, grouped, chunksize=CHUNK_SIZE)):
            detectron2_dicts.append(x)

    return detectron2_dicts


def to_oid_desc(image_id_and_group_df):
    """oid_desc is almost detectron2_dict, except that mask is not in format of polygon or RLE."""

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
    image_id, group_df = image_id_and_group_df
    if 'get-height-width':
        oh, ow = origin_height, origin_width = group_df[['Height', 'Width']].iloc[0]
    if 'get-annotations':
        annotations = []
        for _, row in group_df.iterrows():
            # 'MaskPath' here actually is mask file name(e.g. 88e582a7b14e34a8_m039xj__6133896f.png)
            mask_fname, mid, x1, x2, y1, y2 = \
                row[['MaskPath', 'LabelName', 'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax']]
            mask_id = os.path.splitext(mask_fname)[0]
            annotations.append({'bbox'        : [x1 * ow, y1 * oh, x2 * ow, y2 * oh],
                                'bbox_mode'   : BoxMode.XYXY_ABS,
                                'mask_id'     : mask_id,
                                'category_id' : MID_TO_NO[mid], })

    return {'image_id'   : image_id,
            'file_name'  : f'{image_id}.jpg',  # required by convert_to_coco_dict.
            'height'     : origin_height,      # height / width are implied by image files.
            'width'      : origin_width,       # height / width are implied by image files.
            'annotations': annotations}


def oid_descs_to_detectron2_dicts(descs, masks_dir, cache_path=None):
    """Add Annotation.segmentation fields compliant to detectron2 format."""
    with Pool() as pool:
        dicts = list(tqdm(
            pool.imap(partial(oid_desc_to_detectron2_dict, masks_dir=masks_dir), descs, chunksize=CHUNK_SIZE)
        ))

    if cache_path is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump(dicts, f)

    return dicts


def oid_desc_to_detectron2_dict(desc, masks_dir):
    oh, ow = origin_height, origin_width = desc['height'], desc['width']
    for anno in desc['annotations']:
        mask_path = os.path.join(masks_dir, f'{anno["mask_id"]}.png')
        mask_bin = cv2.resize(cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE), dsize=(ow, oh))
        mask_bin = (mask_bin > MASK_THRESHOLD).astype('uint8')
        mask_rle = pycocotools.mask.encode(np.asarray(mask_bin, order="F"))

        assert (pycocotools.mask.decode(mask_rle) == mask_bin).all()

        anno['segmentation'] = mask_rle

    return desc


def make_mapper(dataset_name, is_train=True, augmentations: T.AugmentationList = None):
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

        if not is_train:
            return {
                "image_id" : desc['image_id'],  # COCOEvaluator.process() need it.
                # expected shape: [C, H, W]
                "image"    : torch.as_tensor(np.ascontiguousarray(auged_image.transpose(2, 0, 1))),
                "height"   : auged_height,
                "width"    : auged_width,
            }

        target = Instances(image_size=(ah, aw))
        if 'fill gt_boxes':
            # shape: n_box, 4
            boxes_abs = np.array([anno['bbox'] for anno in desc['annotations']])
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
            "image_id" : desc['image_id'],  # COCOEvaluator.process() need it.
            # expected shape: [C, H, W]
            "image"    : torch.as_tensor(np.ascontiguousarray(auged_image.transpose(2, 0, 1))),
            "height"   : auged_height,
            "width"    : auged_width,
            "instances": target,  # refer: annotations_to_instances()
        }

    return _desc_to_example


def register_all_oid():
    # register dataset
    for tv in ["train", "validation"]:
        is_train = tv == 'train'
        paths = get_paths(OID_DIR, tv)
        ds_name = "oid_" + tv
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
        # make sure validation dataset compliant to Detectron2 Format.
        if is_train:
            pipeline = DescPipeline([partial(get_oid_descs, is_train=is_train), ],
                                    cache_paths=[paths.descs_pkl, ])
        else:
            pipeline = DescPipeline([partial(get_oid_descs, is_train=is_train),
                                     partial(oid_descs_to_detectron2_dicts, masks_dir=paths.masks_dir, cache_path=paths.dicts_pkl)],
                                    cache_paths=[paths.descs_pkl, paths.dicts_pkl])
        # register oid dataset dicts.
        DatasetCatalog.register("oid_" + tv, pipeline)
        # set oid metadata.
        MetadataCatalog.get(ds_name).set(images_dir=paths.images_dir,
                                         masks_dir=paths.masks_dir,
                                         # json_file=json_file,
                                         image_root=paths.images_dir,
                                         evaluator_type="tfod",
                                         thing_classes=KLASS_NAMES,
                                         no_to_mid=NO_TO_MID)


# ____________________________ Initialization __________________________________
register_all_oid()


if __name__ == "__main__":
    # clear_pkl()
    dt = DatasetCatalog.get('oid_train')
    dv = DatasetCatalog.get('oid_validation')
#     if '--config-file' in sys.argv:
#         CLI_ARGS = sys.argv[1:]
#     else:
#         CLI_ARGS = [
#             '--config-file', DETECTRON2_DIR + '/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
#             '--num-gpus', '1', 'SOLVER.IMS_PER_BATCH', '8', 'SOLVER.BASE_LR', '0.0025',
#             'DATASETS.TRAIN', '("coco_2017_val", )',
#             # '--opts',
#             # 'MODEL.WEIGHTS', './weights/model_final_f10217.pkl',
#             # 'MODEL.DEVICE', 'cpu'
#         ]
#
#     ARGS = default_argument_parser().parse_args(CLI_ARGS)
#
#     if 'setup(args)':
#         args = ARGS
#         cfg = get_cfg()
#         cfg.merge_from_file(args.config_file)
#         cfg.merge_from_list(args.opts)
#         cfg.freeze()
#         default_setup(
#             cfg, args
#         )  # if you don't like any of the default setup, write your own setup code
#
#     if 'build_detection_train_loader':
#
#         class ZipDataset(Dataset):
#             def __init__(self, datasets: List[Dataset]):
#                 self.datasets = datasets
#
#             def __getitem__(self, index):
#                 return list(map(lambda x: x[index], self.datasets))
#
#             def __len__(self):
#                 return min(list(map(lambda x: len(x), self.datasets)))
#
#         if 'get_detection_dataset_dicts':
#             dicts_train: List[Dict] = DatasetCatalog.get("oid_train")
#             dicts_valid: List[Dict] = DatasetCatalog.get("oid_validation")
#             ds_dict = DatasetFromList(dicts_train, copy=False)
#         if 'No_Augmentation':
#             ds_raw = MapDataset(ds_dict, make_mapper('oid_train', ))
#         if 'DatasetMapper':
#             augs = build_augmentation(cfg, is_train=True)
#             ds_aug = MapDataset(ds_dict, make_mapper('oid_train', T.AugmentationList(augs)))
#
#         ds_zip = ZipDataset([ds_raw, ds_aug])
#
#         def make_gen():
#             sampler = TrainingSampler(len(ds_dict))
#             for i in sampler:
#                 print(f'index : {i}')
#                 yield ds_zip[i]
#
#         gen_zip = make_gen()
#
#     if 'visualization':
#         from matplotlib import pyplot as plt
#         from detectron2.utils.visualizer import *
#
#         def draw_masks(vis: Visualizer, ins: Instances):
#             for mask in ins.get_fields()['gt_masks']:
#                 vis.draw_binary_mask(mask.numpy())
#
#         def draw_boxes(vis: Visualizer, ins: Instances):
#             for box in ins.get_fields()['gt_boxes']:
#                 vis.draw_box(box)
#
#         def draw_instances(vis: Visualizer, ins: Instances):
#             f = ins.get_fields()
#             vis.overlay_instances(boxes=f['gt_boxes'],
#                                   labels=f['gt_classes'].tolist(),
#                                   masks=f['gt_masks'])
#
#         def show(visualizer):
#             plt.imshow(visualizer.get_output().get_image())
#             plt.show()
#
#         exam_raw, exam_aug = next(gen_zip)
#         exam_aug['instances']
#
#         v_raw = Visualizer(exam_raw['image'].numpy().transpose(1, 2, 0))
#         show(v_raw)
#         draw_instances(v_raw, exam_raw['instances'])
#         show(v_raw)
#
#         v_aug = Visualizer(exam_aug['image'].numpy().transpose(1, 2, 0))
#         show(v_aug)
#         draw_instances(v_aug, exam_aug['instances'])
#         show(v_aug)
#
#
