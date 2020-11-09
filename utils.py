#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__="Frank Jing"


import os


def dirname_ntimes(path_or_dir, ntimes=1):
    ret = path_or_dir
    for _ in range(ntimes):
        ret = os.path.dirname(ret)
    return ret


def id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]
