import glob
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from multiprocessing import Pool

IMAGES_DIR = "/root/autodl-tmp/gen_img"
CANNY_DIR = "/root/autodl-tmp/gen_img_light_blur"
CANNY2_DIR = "/root/autodl-tmp/gen_img_light"

os.makedirs(CANNY_DIR, exist_ok=True)
os.makedirs(CANNY2_DIR, exist_ok=True)

def process_image(img_file):
    can_file = CANNY_DIR + "/" + os.path.basename(img_file)
    can_file2 = CANNY2_DIR + "/" + os.path.basename(img_file)

    if os.path.exists(can_file):
        print("Skip: " + img_file)
        return

    print(can_file)
    image = Image.open(img_file)

    # 去色处理
    grayscale_image = image.convert('L')
    grayscale_image.save(can_file2)

    # 模糊处理
    blurred_image = image.filter(ImageFilter.GaussianBlur(5))

    # 增加对比度
    enhancer = ImageEnhance.Contrast(blurred_image)
    contrast_image = enhancer.enhance(1.2)  # 这里的参数可以调整，2.0表示增加两倍对比度

    contrast_image.save(can_file)

if __name__ == '__main__':
    img_files = glob.glob(IMAGES_DIR + "/*.png")
    with Pool(processes=4) as pool:  # 这里使用4个进程
        pool.map(process_image, img_files)