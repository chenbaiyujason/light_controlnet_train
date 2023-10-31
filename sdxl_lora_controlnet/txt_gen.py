import glob
import os
from PIL import Image

IMAGES_DIR = "D:\CBY_YZZ\controlnetTrain\gen_txt"

img_files = glob.glob(IMAGES_DIR + "/*.png")
for img_file in img_files:
    cap_file = img_file.replace(".png", ".txt")
    if os.path.exists(cap_file):
        print(f"Skip: {img_file}")
        continue
    print(img_file)

    img = Image.open(img_file)
    prompt = img.text["prompt"] if "prompt" in img.text else ""
    if prompt == "":
        print(f"Prompt not found in {img_file}")

    with open(cap_file, "w") as f:
        f.write(prompt + "\n")