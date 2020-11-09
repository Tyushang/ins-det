# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import torch
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from engine import DefaultPredictor

from predictor import VisualizationDemo

# constants
from utils.visualizer import Visualizer

WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs-oid/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":

    PRJ_DIR = '/data/venv-pytorch/detectron2'

    import platform
    RUN_ON = 'local' if platform.node() == 'frank-note' else 'google-cloud'
    if RUN_ON == 'local':
        os.chdir(PRJ_DIR + '/tyu')
    CLI_ARGS = [
        '--config-file', '../configs-oid/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        '--input', './sample_images/input1.jpg', './sample_images/input2.jpg',
        '--opts',
        'MODEL.WEIGHTS', './weights/model_final_f10217.pkl',
        'MODEL.DEVICE', 'cpu'
    ]

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args(CLI_ARGS)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # predictor is a wrapper of model. use predictor.model to get model.
    # /type model: detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN
    predictor = DefaultPredictor(cfg)
    # weights:
    # import pickle
    # with open(cfg.MODEL.WEIGHTS, 'rb') as f:
    #     weights = pickle.load(f, encoding='latin1')

    if 'predict and visualization':
        for path in args.input:
            raw_image = read_image(path, format="BGR")
            tic       = time.time()
            # prediction
            with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
                height, width = raw_image.shape[:2]
                img = predictor.aug.get_transform(raw_image).apply_image(raw_image)
                # here: image shape should be [C, H, W], and BGR format
                img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

                inputs = {"image": img, "height": height, "width": width}
                pred_all = predictor.model([inputs])
                # {'instances': detectron2.structures.instances.Instances}
                pred     = pred_all[0]
            print(f'time elapsed: {time.time() - tic:.3f}s')

            visualizer = Visualizer(raw_image[:, :, ::-1])
            # noinspection DuplicatedCode
            if "panoptic_seg" in pred:
                ...
            else:
                if "sem_seg" in pred:
                    pred_vis = visualizer.draw_sem_seg(
                        pred["sem_seg"].argmax(dim=0).to(torch.device("cpu"))
                    )
                if "instances" in pred:
                    instances = pred["instances"].to(torch.device("cpu"))
                    pred_vis = visualizer.draw_instance_predictions(predictions=instances)

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, pred_vis.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit

