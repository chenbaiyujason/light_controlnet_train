import os

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter,ImageOps
import numpy as np
import cv2

from datasets import load_dataset,load_from_disk

from pathlib import Path
import wandb
import random


cache_dir = "/mnt/disks/hfcache/deimg"
Path(cache_dir).mkdir(parents=True, exist_ok=True)
odatapath="/mnt/disks/consdata/consandeimg/"
testdatapath="/mnt/disks/testdata/1000/"
Path(cache_dir).mkdir(parents=True, exist_ok=True)

# dataset = load_dataset("ioclab/animesfw", cache_dir=cache_dir,split="train[:1000]")
dataset=load_from_disk(testdatapath)
# dataset.set_transform(transforms)
# dataset.save_to_disk(testdatapath)
print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
# datasettest = datasettest.remove_columns("conditioning_image")
# datasettest = datasettest.map(transforms, batched=True,num_proc=220)
# print(datasettest.column_names)
# dataset = dataset.remove_columns("conditioning_image")
# dataset = dataset.map(transforms,batched=True)
# dataset.save_to_disk(odatapath)
# dataset.push_to_hub('ioclab/lighttest', private=True, max_shard_size="10GB")