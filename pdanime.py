import os
import time

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter,ImageOps
import numpy as np
import cv2

from datasets import load_dataset,load_from_disk

from pathlib import Path
import wandb
import random
from tqdm import tqdm
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
    # print(f"{time.time()}处理图片{img}")
    global inti
    global pbar
    inti +=1
    progress = inti / 400000

    pbar.update(1 / progress / 100)
    pbar.set_description(f"处理数量: n={inti}")
    return img

def transforms(examples):
    examples["conditioning_image"] = [imgprocess(image) for image in examples["image"]]
    return examples


# wandb.init(project='light-dataset-test')
#

pbar = tqdm(total=100, desc="Progress", leave=False,  bar_format="{desc}: {percentage:.3f}%")
inti=0
cache_dir = "/mnt/disks/hfcache/deimg"
# Path(cache_dir).mkdir(parents=True, exist_ok=True)
# odatapath="/mnt/disks/consdata/consandeimg/"
testdatapath="/mnt/disks/testdata/600k/"
Path(testdatapath).mkdir(parents=True, exist_ok=True)
dataset = load_dataset("ioclab/animesfw", cache_dir=cache_dir,split= 'train[400000:600000]')
# dataset=dataset.train_test_split(test_size=0.001, shuffle=True)["test"]


# dataset=load_from_disk("/mnt/disks/hfcache/deimg")
# # num_examples = dataset.num_columns
# # empty_images.fill(None)

# dataset=load_from_disk(testdatapath)
# print(dataset.column_names)
# print( dataset.num_columns)
# print(dataset.num_rows)
# imageblack = Image.new("RGB", (16, 16), color="black")
# dataset=dataset.remove_columns("conditioning_image")
print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
# dataset=dataset.add_column(name="conditioning_image", column=[bytes_array] * dataset.num_rows)

# dataset.save_to_disk(testdatapath)
# dataset.set_transform(transforms)
print(dataset[0])
print("开始map")
# dataset.reset_format()
# # datasettest = datasettest.remove_columns("conditioning_image")
# # datasettest = datasettest.map(transforms, batched=True,num_proc=220)
# # print(datasettest.column_names)
# # dataset = dataset.remove_columns("conditioning_image")
dataset = dataset.map(transforms,batched=True,num_proc=80)
print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
print("处理完成")
# dataset = dataset.rename_column("my_subset", "train")
# dataset.save_to_disk(odatapath)
dataset.save_to_disk(testdatapath)
print("保存完成")
# dataset.push_to_hub('ioclab/lighttestout', private=False, max_shard_size="1GB")