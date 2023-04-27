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
    # print(f"{time.time()}处理图片{img}")
    global inti
    global pbar
    inti +=1
    progress = inti / 400000

    pbar.update(1 / progress / 100)
    pbar.set_description(f"处理数量: n={inti}")
    return img


def oimgprocess(image):
    rand_num = random.uniform(0.1, 0.3)
    rand_num = round(rand_num, 2)
    print(rand_num)


    if image.mode != 'RGB':
        image = image.convert('RGB')

    enhancement_type = random.randint(1, 3)

    # 获取亮部区域和暗部区域的范围
    if enhancement_type == 1:
        # 增加曝光度
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(1 + rand_num)
    elif enhancement_type == 2:
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = 1.2 + rand_num * 0.5  # 增强因子
        image = enhancer.enhance(brightness_factor)
        enhanced_image = image
    else:
        im = levels(image, black=50 + random.randint(0, 30), white=255, gamma=1)
        enhanced_image = im
    return enhanced_image


def levels(image, black=0, white=255, gamma=1.0):
    # 灰场调整
    if black == white:
        return image.point(lambda x: black)

    # 计算变换参数
    in_black = float(min(black, white))
    in_white = float(max(black, white))
    in_scale = in_white - in_black
    out_black = 0.0
    out_white = 255.0
    out_scale = out_white - out_black

    if in_scale < 0.0001:
        return image

    gamma_inv = 1.0 / gamma if gamma > 0.0 else 1.0

    def adjust(x):
        if x <= in_black:
            return 0.0
        elif x >= in_white:
            return 255.0
        else:
            normalized = (x - in_black) / in_scale
            exponent = pow(normalized, gamma_inv)
            return (out_scale * exponent) + out_black

    return image.point(adjust)


def transforms(examples):
    examples["conditioning_image"] = [imgprocess(image) for image in examples["newo_image"]]
    return examples

def ntransforms(examples):
    examples["newo_image"] = [oimgprocess(image) for image in examples["image"]]
    return examples
# wandb.init(project='light-dataset-test')
#


cache_dir = "/mnt/disks/consdata/consdeimg"
Path(cache_dir).mkdir(parents=True, exist_ok=True)
odatapath="/mnt/disks/consdata/consdeimgpow/"
# dataset = load_dataset("/mnt/disks/consdata/consdeimg", cache_dir=cache_dir)
dataset=load_from_disk("/mnt/disks/consdata/consdeimg")

print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
# datasettest = datasettest.remove_columns("conditioning_image")
# datasettest = datasettest.map(transforms, batched=True,num_proc=220)
# print(datasettest.column_names)
dataset = dataset.remove_columns("conditioning_image")
dataset = dataset.map(ntransforms, batched=True,num_proc=140)
dataset = dataset.remove_columns("image")
dataset = dataset.map(transforms, batched=True,num_proc=140)
print(dataset.column_names)
print(dataset.num_columns)
print(dataset.num_rows)
dataset.save_to_disk(odatapath)
dataset.push_to_hub('ioclab/lighdatapow', private=False, max_shard_size="1GB")