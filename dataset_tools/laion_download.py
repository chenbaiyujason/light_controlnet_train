# encoding="UTF-8"

from img2dataset import download
import shutil
import multiprocessing
def main():
    download(
        processes_count=16,
        thread_count=64,
        url_list="/mnt/disks/persist/laion2B-en-aesthetic",
        resize_mode="center_crop",
        image_size=512,
        output_folder="/mnt/disks/persist/images/laion2B-en-aesthetic",
        output_format="files",
        input_format="parquet",
        skip_reencode=True,
        save_additional_columns=["similarity","hash","punsafe","pwatermark","aesthetic"],
        url_col="URL",
        caption_col="TEXT",
        distributor="multiprocessing",
        enable_wandb=True,
        wandb_project="tpu-laion-download",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
