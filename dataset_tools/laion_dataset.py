import os

from datasets import load_dataset
from pathlib import Path
from PIL import Image


cache_dir = "/mnt/disks/persist/datasets/cache"
Path(cache_dir).mkdir(parents=True, exist_ok=True)

save_dir = "/mnt/disks/persist/datasets/grayscale_image_aesthetic_3M"
Path(save_dir).mkdir(parents=True, exist_ok=True)


dataset = load_dataset("/mnt/disks/persist/datasets/3m/", cache_dir=cache_dir )
dataset.save_to_disk(save_dir, max_shard_size="1GB")
dataset.push_to_hub('ioclab/grayscale_image_aesthetic_3M', private=True, max_shard_size="1GB")