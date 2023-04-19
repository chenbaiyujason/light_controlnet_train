# -*- coding: utf-8 -*-
import os

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter,ImageOps
import numpy as np
import cv2
import datasets
from danbooru2022 import DanbooruDataset
from datasets import load_dataset
from datasets import load_dataset_builder
from pathlib import Path
import wandb
# 更改此路径为你希望将数据集下载到的目录
custom_cache_dir = "/mnt/disks/hfcache"
# Path(custom_cache_dir).mkdir(parents=True, exist_ok=True)
# os.environ["HF_DATASETS_CACHE"] = custom_cache_dir
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
    img = img.filter(ImageFilter.GaussianBlur(radius=10))

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
    img = img.resize((512, 512), resample=Image.BILINEAR)

    return img

def transforms(examples):
    examples["conditioning_image"] = [imgprocess(image) for image in examples["image"]]
    return examples
# 选择子集，将 '0-sfw' 更改为 '1-full' 或 '2-tags' 以下载其他子集
builder = DanbooruDataset(config_name='0-sfw')

# # 下载数据集
# print("正在下载数据集...")
builder.download_and_prepare(output_dir=custom_cache_dir)

# # 加载数据集
# print("正在加载数据集...")
dataset = builder.as_dataset(split= 'train' )
dataload = "/mnt/disks/hfcache/"
cache_dir = "/mnt/disks/cache/animgsfw"
# dataset = load_dataset(dataload, cache_dir=cache_dir)
# 显示一些数据集信息
print("数据集信息：")
print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
odatapath="/mnt/disks/consdata/consanimeimg/"
dataset = dataset.remove_columns("post_id")
print("移除postid：")
print(dataset)
print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
dataset.push_to_hub('ioclab/animesfw', private=True, max_shard_size="1GB")
dataset.save_to_disk(odatapath)
# cache_dir = "/mnt/disks/data/cache/deanimeimg"
# Path(cache_dir).mkdir(parents=True, exist_ok=True)
# odatapath="/mnt/disks/consdata/consanimeimg/"
# dataset = load_dataset("/mnt/disks/data/grayscale_image_aesthetic_3M/data/", cache_dir=cache_dir)

# datasettest = datasettest.remove_columns("conditioning_image")
# datasettest = datasettest.map(transforms, batched=True,num_proc=220)
# print(datasettest.column_names)
print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
dataset = dataset.map(transforms, batched=True,num_proc=220)
odatapath="/mnt/disks/consdata/consanimelightimg/"
dataset.save_to_disk(odatapath)
dataset.push_to_hub('ioclab/lightanimesfw', private=True, max_shard_size="1GB")

print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
