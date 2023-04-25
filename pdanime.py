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

def imgprocess(img):
    rand_num = random.uniform(-0.3, 1)
    rand_num = round(rand_num, 2)
    # 打开原始图像
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.convert('L')
    # img = ImageOps.equalize(img, mask=None)
    img = img.resize((1536, 1536), resample=Image.BILINEAR)
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(1.3)
    img = img_contrast

    # 使用高斯模糊去除一部分细节
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    img = img.filter(ImageFilter.BoxBlur(radius=15))
    img = img.filter(ImageFilter.GaussianBlur(radius=10+rand_num))

    # 使用中值滤波器去除更多的细节
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SMOOTH)
    # 使用边缘增强滤波器保留光影和色块的模糊关系

    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))

    img_array = np.array(img)

    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 对图像进行自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)

    # 将图像转换为 Pillow 格式，并显示处理后的图像
    img = Image.fromarray(cl_img)
    img = img.resize((img.size[0] // 16, img.size[1] // 16), resample=Image.BOX)
    img = img.resize((img.size[0] * 16, img.size[1] * 16), resample=Image.NEAREST)
    img = img.filter(ImageFilter.GaussianBlur(radius=10))
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(1.2 + rand_num * 0.15)
    img = img_contrast
    img = img.resize((768,768), resample=Image.BILINEAR)

    return img

def transforms(examples):
    examples["conditioning_image"] = [imgprocess(image) for image in examples["image"]]
    return examples


# wandb.init(project='light-dataset-test')
#



cache_dir = "/mnt/disks/hfcache/deimg"
Path(cache_dir).mkdir(parents=True, exist_ok=True)
odatapath="/mnt/disks/consdata/consandeimg/"
testdatapath="/mnt/disks/testdata/1000/"
Path(cache_dir).mkdir(parents=True, exist_ok=True)
dataset = load_dataset("./mnt/disks/hfcache/deimg/ioclab/animesfw", cache_dir=cache_dir,split="train[:1000]")
# dataset=load_from_disk("/mnt/disks/hfcache/deimg")
# # num_examples = dataset.num_columns
# # empty_images.fill(None)
print(dataset.column_names)
print( dataset.num_columns)
print(dataset.num_rows)
# dataset=dataset.add_column(name="conditioning_image", column=[np.array([])] * 1000)
# print(dataset.column_names)
# print(dataset.num_columns)
# print(dataset.num_rows)
# dataset.save_to_disk(testdatapath)
# dataset.set_transform(transforms)
# # datasettest = datasettest.remove_columns("conditioning_image")
# # datasettest = datasettest.map(transforms, batched=True,num_proc=220)
# # print(datasettest.column_names)
# # dataset = dataset.remove_columns("conditioning_image")
# # dataset = dataset.map(transforms,batched=True)
# print(dataset.column_names)
# print(dataset.num_columns)
# print(dataset.num_rows)
# # dataset.save_to_disk(odatapath)
# dataset=load_from_disk(testdatapath)
#
# dataset.push_to_hub('ioclab/lighttestout', private=False, max_shard_size="1GB")