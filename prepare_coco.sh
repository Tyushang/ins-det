#!/bin/bash -e

cd /home/jupyter/datasets/coco
# wget images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
# wget annotations and info (.json format)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# unzip images
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q test2017.zip
# unzip annotations and info (.json format)
unzip annotations_trainval2017.zip
unzip image_info_test2017.zip

