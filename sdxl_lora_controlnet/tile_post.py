import glob
import os
import numpy as np
from PIL import Image, ImageFilter
import numpy.matlib

IMAGES_DIR = "D:\Download\\tt"
CANNY_DIR = "D:\Download\\ttc"


def Sur_blur(I_in, thre, half_size):
    I_out = I_in * 1.0
    row, col = I_in.shape
    w_size = half_size * 2 + 1
    for ii in range(half_size, row - 1 - half_size):
        for jj in range(half_size, col - 1 - half_size):
            aa = I_in[
                ii - half_size : ii + half_size + 1, jj - half_size : jj + half_size + 1
            ]
            p0 = I_in[ii, jj]
            mask_1 = numpy.matlib.repmat(p0, w_size, w_size)
            mask_2 = 1 - abs(aa - mask_1) / (2.5 * thre)
            mask_3 = mask_2 * (mask_2 > 0)
            t1 = aa * mask_3
            I_out[ii, jj] = t1.sum() / mask_3.sum()

    return I_out


os.makedirs(CANNY_DIR, exist_ok=True)
img_files = glob.glob(IMAGES_DIR + "/*.png")
for img_file in img_files:
    can_file = CANNY_DIR + "/" + os.path.basename(img_file)
    if os.path.exists(can_file):
        print("Skip: " + img_file)
        continue

    print(can_file)
    
    img = np.array(Image.open(img_file))
    img_out = img * 1.0
    thre = 10
    half_size = 15
    img_out[:, :, 0] = Sur_blur(img[:, :, 0], thre, half_size)
    img_out[:, :, 1] = Sur_blur(img[:, :, 1], thre, half_size)
    img_out[:, :, 2] = Sur_blur(img[:, :, 2], thre, half_size)

    img_out = img_out / 255

    image_out = Image.fromarray((img_out * 255).astype(np.uint8))
    blurred_image = image_out.filter(ImageFilter.GaussianBlur(radius=3))
    blurred_img = np.array(blurred_image)
    thre = 20
    half_size = 25
    img_out[:, :, 0] = Sur_blur(blurred_img[:, :, 0], thre, half_size)
    img_out[:, :, 1] = Sur_blur(blurred_img[:, :, 1], thre, half_size)
    img_out[:, :, 2] = Sur_blur(blurred_img[:, :, 2], thre, half_size)
    image_out = Image.fromarray((img_out).astype(np.uint8))

    image_out.save(can_file)
