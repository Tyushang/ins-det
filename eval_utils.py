#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"
import contextlib
import copy
import io
import itertools
import json
import logging
import os
from datetime import datetime
from typing import *

import numpy as np
import pandas as pd
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators
from detectron2.evaluation.coco_evaluation import instances_to_coco_json, _evaluate_predictions_on_coco, COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import create_small_table
from fvcore.common.file_io import PathManager
from object_detection.metrics.oid_challenge_evaluation import _load_labelmap
from object_detection.utils import object_detection_evaluation
from pycocotools.coco import COCO
from tabulate import tabulate

from object_detection.metrics import oid_challenge_evaluation_utils as tfod_utils

# noinspection PyMethodMayBeStatic,PyAttributeOutsideInit
from prepare_oid import NO_TO_MID, get_paths, OID_DIR, MASK_THRESHOLD
from utils import encode_binary_mask, encode_mask_from_path


logger = logging.getLogger('detectron2')


# noinspection PyPep8Naming,PyMethodMayBeStatic,PyAttributeOutsideInit
class MyCocoEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, *, use_fast_impl=True):
        self._predictions = []
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger('MyCocoEvaluator')

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        return tasks

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            prediction = {"image_id": inp["image_id"]}

            # TODO this is ugly
            if "instances" in out:
                instances = out["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, inp["image_id"])
            if "proposals" in out:
                prediction["proposals"] = out["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                        category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


class TfodEvaluator(DatasetEvaluator):

    def __init__(self, oid_paths, no_to_mid, distributed=True):
        self.logger       = logging.getLogger('TfodEvaluator')
        self._cpu_device  = torch.device("cpu")
        self._distributed = distributed

        self.oid_paths = oid_paths
        self.no_to_mid = no_to_mid

        self.label_map, self.categories = _load_labelmap(oid_paths.label_map_pbtxt)

        self.gt_bboxes       = pd.read_csv(oid_paths.bbox_csv)
        self.gt_image_labels = pd.read_csv(oid_paths.image_label_csv).rename(columns={'Confidence': 'ConfidenceImageLabel'})
        self.gt_masks        = self.get_gt_masks()

        # detectron2 predictions with "image_id" key.
        self.all_ptd2_preds  = []

    def get_gt_masks(self):
        # use cache file to accelerate.
        if os.path.exists(self.oid_paths.tfod_mask_pkl):
            return pd.read_pickle(self.oid_paths.tfod_mask_pkl)
        if 'join image height/width and encode mask':
            # columns: ['ImageID', 'Width', 'Height']
            mask_df  = pd.read_csv(self.oid_paths.mask_csv)
            shape_df = pd.read_csv(self.oid_paths.shape_csv)
            joined   = mask_df.set_index('ImageID').join(shape_df.set_index('ImageID'))
            # MaskPath here is actually mask filename. Fuck!
            mask_paths = joined['MaskPath'].map(lambda filename: os.path.join(self.oid_paths.masks_dir, filename))
            joined['Mask'] = list(map(lambda p, h, w, t=MASK_THRESHOLD: encode_mask_from_path(p, h, w, t),
                                      mask_paths, joined['Height'], joined['Width']))
        joined = joined.reset_index().rename(columns={'BoxXMin': 'XMin',
                                                      'BoxXMax': 'XMax',
                                                      'BoxYMin': 'YMin',
                                                      'BoxYMax': 'YMax',
                                                      'Height' : 'ImageHeight',
                                                      'Width'  : 'ImageWidth'})
        joined.to_pickle(self.oid_paths.tfod_mask_pkl)

        return joined

    def reset(self):
        self.all_ptd2_preds = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            self.all_ptd2_preds.append({
                "image_id" : inp["image_id"],
                "instances": out["instances"].to(self._cpu_device)
            })

    def evaluate(self):
        if 'self._distributed':
            # TODO: error raised when there's no predictions!
            comm.synchronize()
            raw_preds = comm.gather(self.all_ptd2_preds, dst=0)
            raw_preds = list(itertools.chain(*raw_preds))

            if not comm.is_main_process():
                return {}
        else:
            raw_preds = self.all_ptd2_preds

        # TODO: does removing images with null predictions affect metrics?
        if 'filter images with predictions.':
            raw_preds = list(filter(lambda x: len(x['instances']) > 0, raw_preds))
            if len(raw_preds) == 0:
                print("_"*60 + "There's no predictions.")
                return

        image_ids = [x['image_id'] for x in raw_preds]
        all_location_annotations = self.gt_bboxes[self.gt_bboxes['ImageID'].isin(image_ids)]
        all_label_annotations    = self.gt_image_labels[self.gt_image_labels['ImageID'].isin(image_ids)]

        is_instance_segmentation_eval = True
        if 'instance-segmentation-task':
            all_segm_annotations = self.gt_masks[self.gt_masks['ImageID'].isin(image_ids)]
            # Note: this part is unstable as it requires the float point numbers in both
            # csvs are exactly the same;
            # Will be replaced by more stable solution: merge on LabelName and ImageID
            # and filter down by IoU.
            all_location_annotations = pd.merge(all_location_annotations, all_segm_annotations,
                                                how='outer',
                                                on=['LabelName', 'ImageID', 'XMin', 'XMax', 'YMin', 'YMax',])

        all_annotations = pd.concat([all_location_annotations, all_label_annotations])

        class_label_map, categories = self.label_map, self.categories
        challenge_evaluator = (
            object_detection_evaluation.OpenImagesChallengeEvaluator(
                categories, evaluate_masks=is_instance_segmentation_eval))

        all_predictions = ptd2_preds_to_tfod_eval_preds(
            raw_preds, image_ids=image_ids, no_to_mid=self.no_to_mid)

        images_processed = 0
        for _, groundtruth in enumerate(all_annotations.groupby('ImageID')):
            image_id, image_groundtruth = groundtruth
            groundtruth_dictionary = tfod_utils.build_groundtruth_dictionary(
                image_groundtruth, class_label_map)
            challenge_evaluator.add_single_ground_truth_image_info(
                image_id, groundtruth_dictionary)
            prediction_dictionary = tfod_utils.build_predictions_dictionary(
                all_predictions.loc[all_predictions['ImageID'] == image_id],
                class_label_map)
            # TODO: this will call compute_object_detection_metrics(), which has high time-consumption.
            challenge_evaluator.add_single_detected_image_info(image_id,
                                                               prediction_dictionary)
            images_processed += 1

        metrics = challenge_evaluator.evaluate()

        # with open(FLAGS.output_metrics, 'w') as fid:
        #     io_utils.write_csv(fid, metrics)

        return OrderedDict({'instance-segmentation': metrics})


def get_evaluator2(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == 'oid':
        evaluator_list.append(MyCocoEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == 'tfod':
        paths = get_paths(OID_DIR, 'validation')
        evaluator_list.append(TfodEvaluator(paths, MetadataCatalog.get(dataset_name).no_to_mid, distributed=True))

    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def ptd2_preds_to_tfod_eval_preds(preds, image_ids, no_to_mid=NO_TO_MID):
    """
    - Model(ptd2 maskrcnn) Output Format
        When in inference mode, the builtin models output a list[dict], one dict for each image. Based on the tasks the model is doing, each dict may contain the following fields:
        instances   : Instances object with the following fields:
            pred_boxes      : Boxes object storing N boxes, one for each detected instance.
            scores          : Tensor, a vector of N confidence scores.
            pred_classes    : Tensor, a vector of N labels in range [0, num_categories).
            pred_masks      : a Tensor of shape (N, H, W), masks for each detected instance.
        sem_seg     : ...
    - tfod evaluation predication csv format
        | ImageID | ImageWidth | ImageHeight | LabelName | Score | Mask |
        where mask is: None entries, or strings with base64, zlib compressed, COCO RLE-encoded binary masks.
        注：ImageID is not unique key.
    @param no_to_mid:
    @param image_ids:
    @param preds:
    @return:
    """
    pred_dicts = []

    instances_list = [p['instances'] for p in preds]
    for image_id, ins in zip(image_ids, instances_list):  # foreach image instances:
        origin_image_height, origin_image_width = ins.image_size
        for klass, score, mask in zip(ins.pred_classes, ins.scores, ins.pred_masks):
            pred_dicts.append({
                'ImageID'    : image_id,
                'ImageWidth' : origin_image_width,
                'ImageHeight': origin_image_height,
                'Score'      : float(score),
                'Mask'       : encode_binary_mask(mask.numpy()),
                'LabelName'  : no_to_mid[int(klass)],
            })

    return pd.DataFrame.from_records(pred_dicts)


def ptd2_preds_to_kaggle_eval_preds(preds, image_ids, no_to_mid=NO_TO_MID):
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
        encoded_mask = encode_binary_mask(mask.numpy())
        return f"{mid} {confidence} {encoded_mask}"

    pred_dicts = []
    instances_list = [p['instances'] for p in preds]
    for image_id, ins in zip(image_ids, instances_list):  # foreach image instances:
        origin_image_height, origin_image_width = ins.image_size
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






