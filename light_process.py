import os

from datasets import Dataset
from pathlib import Path
from PIL import Image

import jsonlines

data_dir = Path(r"/mnt/disks/data/grayscale_image_aesthetic_3M")


output_image_foder = Path("/mnt/disks/data/consdata/deimg")
Path(output_image_foder).mkdir(parents=True, exist_ok=True)

output_conditioning_image_foder = Path("/mnt/disks/data/consdata/deconimg")
Path(output_conditioning_image_foder).mkdir(parents=True, exist_ok=True)


def entry_for_id(raw_image_dir, index):
    img = Image.open(raw_image_dir)
    processed_image = img.convert('L')
    caption_dir = f"{raw_image_dir}".replace('.jpg', '.txt')

    with open(caption_dir) as f:
        caption = f.read()

    output_image_dir = output_image_foder / f"{index}.jpg"
    output_conditioning_dir = output_conditioning_image_foder / f"{index}.jpg"

    img.save(output_image_dir)
    processed_image.save(output_conditioning_dir)

    # write to meta.jsonl
    meta = {
        "image": f"{output_image_dir}",
        "conditioning_image": f"{output_conditioning_dir}",
        "caption": caption,
    }
    with jsonlines.open(
            f"/mnt/disks/persist/datasets/3m/meta.jsonl", "a"
    ) as writer:  # for writing
        writer.write(meta)


max_images = 3000000


def generate_entries():
    index = 0

    image_folders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for image_folder in image_folders:

        image_folder = Path(image_folder)
        print(image_folder)

        for filename in os.listdir(image_folder):
            if not filename.endswith('.jpg'):
                continue

            try:

                entry_for_id(image_folder / filename, index)

                if index % 10000 == 0:
                    print(index)

                index += 1
                if index >= max_images:
                    break

            except Exception as e:
                continue

        if index >= max_images:
            break


generate_entries()

# cache_dir = "/mnt/disks/persist/datasets/cache"
# Path(cache_dir).mkdir(parents=True, exist_ok=True)
# ds = Dataset.from_generator(generate_entries, cache_dir=cache_dir)

# save_dir = "/mnt/disks/persist/datasets/grayscale_image_aesthetic_3M"
# Path(save_dir).mkdir(parents=True, exist_ok=True)
# ds.save_to_disk(save_dir)

# ds.push_to_hub('ioclab/grayscale_image_aesthetic_3M', private=True)