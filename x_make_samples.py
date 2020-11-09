#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"


import os
import pandas as pd

# tv: train or validation
tv = 'train'

# Get Sample ids
urls_path = f'../open-images-dataset/{tv}-samples/urls.txt'
ids = list(map(lambda s: os.path.basename(s).split('.')[0],  open(urls_path, 'r').readlines()))

# Select mask sample anno by ids
mask_path = f'../open-images-dataset/annotation-instance-segmentation/{tv}/challenge-2019-{tv}-masks/challenge-2019-{tv}-segmentation-masks.csv'
mask_df   = pd.read_csv(mask_path)
select = pd.Series([True if i in ids else False for i in mask_df['ImageID']])
mask_sample_df = mask_df[select]

# Save mask sample anno
mask_sample_path = os.path.dirname(mask_path) + f'/challenge-2019-{tv}-segmentation-masks-samples.csv'
mask_sample_df.to_csv(mask_sample_path, index=False)
