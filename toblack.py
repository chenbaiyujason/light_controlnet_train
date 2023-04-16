import os

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter,ImageOps
import numpy as np
import cv2


def imgprocess(path):
    # 打开原始图像
    img = Image.open(path)
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
    img= img.filter(ImageFilter.SMOOTH)
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


# 保存处理后的图像
# img.save('filtered_image.jpg')
if __name__ == '__main__':
    files = os.listdir()
    for filename in files:
        # 判断当前文件是否为图片文件
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # 拼接文件路径
            filepath = os.path.join(os.getcwd(), filename)

            # 调用函数 A 进行处理
            processed_img = imgprocess(filepath)

            # 显示处理后的图像
            fig, ax = plt.subplots(1, 2,dpi=300)
            ax[1].imshow(np.array(processed_img.convert('RGB')))
            ax[0].set_title('Original Image')
            ax[0].imshow(Image.open(filepath))
            ax[1].set_title('Processed Image')
            plt.show()