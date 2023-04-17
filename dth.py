import os

from datasets import load_dataset
from pathlib import Path
from PIL import Image


# cache_dir = "/mnt/disks/persist/datasets/cache"
# Path(cache_dir).mkdir(parents=True, exist_ok=True)

# save_dir = "/mnt/disks/persist/datasets/grayscale_image_aesthetic_3M"
# Path(save_dir).mkdir(parents=True, exist_ok=True)

cache_dir = "/mnt/disks/data/cache/deimg"
odatapath="/mnt/disks/data/consdata/consdeimg/"
dataset = load_dataset(odatapath, cache_dir=cache_dir )
# dataset.save_to_disk(save_dir, max_shard_size="1GB")
dataset.push_to_hub('ioclab/light', private=True, max_shard_size="1GB")