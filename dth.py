import os

from datasets import load_dataset
from datasets import load_from_disk
from pathlib import Path
from PIL import Image


# cache_dir = "/mnt/disks/persist/datasets/cache"
# Path(cache_dir).mkdir(parents=True, exist_ok=True)

# save_dir = "/mnt/disks/persist/datasets/grayscale_image_aesthetic_3M"
# Path(save_dir).mkdir(parents=True, exist_ok=True)

cache_dir = "/mnt/disks/data/cache/consdeimg"
Path(cache_dir).mkdir(parents=True, exist_ok=True)
odatapath="/mnt/disks/data/consdata/consdeimg/train"
dataset = load_from_disk(odatapath )
# dataset.save_to_disk(save_dir, max_shard_size="1GB")
dataset.push_to_hub('ioclab/light', private=True, max_shard_size="1GB")