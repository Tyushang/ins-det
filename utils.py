#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"


import os

import base64
import numpy as np
# noinspection PyProtectedMember
from cv2 import cv2
from pycocotools import _mask as coco_mask
import typing as t
import zlib


def dirname_ntimes(path_or_dir, ntimes=1):
    ret = path_or_dir
    for _ in range(ntimes):
        ret = os.path.dirname(ret)
    return ret


def id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def encode_mask_from_path(mask_path, height, width, threshold=127):
    mask_bin = cv2.resize(cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE), dsize=(width, height))
    mask_bin = (mask_bin > threshold)
    return encode_binary_mask(mask_bin)












