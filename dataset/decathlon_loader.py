import os
from typing import Any, Optional

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import torch


SET_TYPES = ["train", "validation", "test"]
INPUT_TYPES = ["original", "features"]

id_list = ["imagenet", "cifar100", "aircraft", "daimlerpedcls", "dtd", "gtsrb", "omniglot", "svhn", "ucf101", "vgg-flowers"]

xception_tf_pipeline = transforms.Compose(
    [
        transforms.Resize(
            size=333,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias=None,
        ),
        transforms.CenterCrop(size=(299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

class Decathlon_Dataset(Dataset):
    # using val instead of validation/test, weird!
    def __init__(self, task_id, set_type, input_type, full_dataset=True):

        if not full_dataset:
            assert input_type == "features" and set_type == "train", "This is only for GMMC and MAHA training, \
                                                                and not even for GMMC and MAHA validation,\
                                                                use this carefully."
        super().__init__()
        """init function for the dataset"""
        self.pipeline = xception_tf_pipeline
        self.dataset_name = id_list[task_id]
        self.set_type = set_type
        self.input_type = input_type
        if task_id != 0:
            self.path = os.path.join("/lab/tmpig15b/u/Decathlon", id_list[task_id])
            self.label_file = os.path.join(self.path, f"{set_type}_labels.txt")
            self.label_diff = 1
        else:
            self.label_file = f"/lab/harry/WorkSpace/yuecheng_code/dataset/Imagenet_{self.set_type}_label.txt"
            self.label_diff = 0



        self.image_paths = []
        self.labels = []
        self.label_dict = {}
        with open(self.label_file, "r") as f:
            for line in f.readlines():
                image_path, label = line.strip().split(",")
                label = int(label) - self.label_diff
                if label not in self.label_dict:
                    self.label_dict[label] = label 
                self.image_paths.append(image_path)
                self.labels.append(label)
        self.num_classes = len(self.label_dict.keys())

        if task_id == 0:
            if full_dataset:
                self.feature = torch.load(f"/lab/tmpig15b/u/Decathlon/imagenet/xception_{self.set_type}_features_orig.pth", map_location="cpu").cpu()
            else:
                self.feature = torch.load(f"/lab/harry/WorkSpace/yuecheng_data/imagenet_mini/train_feature/xception_feature.pth", map_location="cpu").cpu()
                self.labels = torch.load(f"/lab/harry/WorkSpace/yuecheng_data/imagenet_mini/train_feature/xception_label.pth", map_location="cpu").cpu()
                print("please note most of the part will be deprecated and only .num_classes, .dataset_name, and loader will work")
        else:
            self.feature = torch.load(os.path.join("/lab/tmpig15b/u/Decathlon", self.dataset_name, 
                                                    f"xception_{self.set_type}_features", 
                                                    set_type+"_features.pth"))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple:

        if self.input_type == "features":
            return self.feature[index], self.labels[index]
        else:
            image_filename = self.image_paths[index]
            with Image.open(image_filename) as image:
                # for the original images, make sure to convert to RGB
                image = image.convert("RGB")
            image_tensor = self.pipeline(image)
            return image_tensor, self.labels[index]


if __name__ == "__main__":

    pass