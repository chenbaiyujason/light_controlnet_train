
from datasets import load_dataset
odatapath="/mnt/disks/data/grayscale_image_aesthetic_3M/"
dataset = load_dataset("/mnt/disks/data/grayscale_image_aesthetic_3M/data/", split="train")


def transforms(examples):
    examples["conditioning_imag"] = [image.convert("RGB").resize((100,100)) for image in examples["image"]]
    return examples

dataset = dataset.map(transforms, remove_columns=["conditioning_image"], batched=True)