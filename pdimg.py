import os

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter,ImageOps
import numpy as np
import cv2

from datasets import load_dataset




def imgprocess(img):
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
    img_contrast = enhancer.enhance(1.3)
    img = img_contrast
    img = img.resize((512, 512), resample=Image.BILINEAR)

    return img


def transforms(examples):
    examples["conditioning_image"] = [imgprocess(image) for image in examples["image"]]
    return examples

odatapath="/mnt/disks/data/consdata/consdeimg/"
dataset = load_dataset("/mnt/disks/data/grayscale_image_aesthetic_3M/data/", split="train")
dataset = dataset.map(transforms, remove_columns=["conditioning_image"], batched=True,num_proc=120)
dataset.save_to_disk('odatapath')