import os
import datasets
from huggingface_hub import HfApi
from datasets import DownloadManager, DatasetInfo
from datasets.data_files import DataFilesDict

_EXTENSION = [".png", ".jpg", ".jpeg"]
_NAME = "animelover/danbooru2022"
_REVISION = "main"


class DanbooruDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        # add number before name for sorting
        datasets.BuilderConfig(
            name="0-sfw",
            description="sfw subset",
        ),
        datasets.BuilderConfig(
            name="1-full",
            description="full dataset",
        ),
        datasets.BuilderConfig(
            name="2-tags",
            description="only tags of dataset",
        ),
    ]

    def _info(self) -> DatasetInfo:
        if self.config.name == "2-tags":
            features = {
                "tags": datasets.Value("string"),
                "post_id": datasets.Value("int64")
            }
        else:
            features = {
                "image": datasets.Image(),
                "tags": datasets.Value("string"),
                "post_id": datasets.Value("int64")
            }
        return datasets.DatasetInfo(
            description=self.config.description,

            features=datasets.Features(features),
            supervised_keys=None,
            citation="",
        )

    def _split_generators(self, dl_manager: DownloadManager):
        hfh_dataset_info = HfApi().dataset_info(_NAME, revision=_REVISION, timeout=100.0)
        data_files = DataFilesDict.from_hf_repo(
            {datasets.Split.TRAIN: ["**"]},
            dataset_info=hfh_dataset_info,
            allowed_extensions=["zip"],
        )
        gs = []
        for split, files in data_files.items():
            downloaded_files = dl_manager.download_and_extract(files)
            gs.append(datasets.SplitGenerator(name=split, gen_kwargs={"filepath": downloaded_files}))
        return gs

    def _generate_examples(self, filepath):
        for path in filepath:
            all_fnames = {os.path.relpath(os.path.join(root, fname), start=path)
                          for root, _dirs, files in os.walk(path) for fname in files}
            image_fnames = sorted([fname for fname in all_fnames if os.path.splitext(fname)[1].lower() in _EXTENSION],
                                  reverse=True)
            for image_fname in image_fnames:
                image_path = os.path.join(path, image_fname)
                tags_path = os.path.join(path, os.path.splitext(image_fname)[0] + ".txt")
                with open(tags_path, "r", encoding="utf-8") as f:
                    tags = f.read()
                if self.config.name == "0-sfw" and any(tag.strip() in nsfw_tags for tag in tags.split(",")):
                    continue
                post_id = int(os.path.splitext(os.path.basename(image_fname))[0])
                if self.config.name == "2-tags":
                    yield image_fname, {"tags": tags, "post_id": post_id}
                else:
                    yield image_fname, {"image": image_path, "tags": tags, "post_id": post_id}


nsfw_tags = ["nude", "completely nude", "topless", "bottomless", "sex", "oral", "fellatio gesture", "tentacle sex",
             "nipples", "pussy", "vaginal", "pubic hair", "anus", "ass focus", "penis", "cum", "condom", "sex toy"]