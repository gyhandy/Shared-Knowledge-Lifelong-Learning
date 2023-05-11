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

MNIST_Transform =transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

MNIST_Flatten_Transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

Cifar_transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CustomDataset(Dataset):
    def __init__(self, dataset_name, task_id, set_type, input_type,
                 base_path="/lab/tmpig8d/u/ax-data/", feature_base_path="/lab/tmpi1b/u/ax-data/"):

        super().__init__()
        """init function for the dataset"""
        self.path = os.path.join(base_path, dataset_name, "task"+str(task_id))
        self.dataset_name = dataset_name+"_"+str(task_id)
        self.set_type = set_type
        self.input_type = input_type
        self.label_file = os.path.join(self.path, f"{set_type}_labels.txt")
        if dataset_name == "mnist":
            self.pipeline = MNIST_Transform
        elif dataset_name == "cifar100":
            self.pipeline = Cifar_transform

        self.image_paths = []
        self.labels = []
        self.label_dict = {}
        with open(self.label_file, "r") as f:
            for line in f.readlines():
                image_path, label = line.strip().split(",")
                label = int(label)
                if label not in self.label_dict:
                    self.label_dict[label] = label 
                self.image_paths.append(image_path)
                self.labels.append(label)
        self.num_classes = len(self.label_dict.keys())

        # self.feature = torch.from_numpy(np.load(os.path.join(feature_base_path, dataset_name, 
        #                                             "task"+str(task_id), "resnet18"+"_features", 
        #                                             set_type+"_features.npy"))).float()
        self.feature = torch.load(os.path.join(feature_base_path, dataset_name, 
                                                "task"+str(task_id), "resnet18"+"_features", 
                                                set_type+"_features.pth"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple:

        if self.input_type == "features":
            return self.feature[index], self.labels[index]
        else:
            if self.dataset_name[:5] == "mnist" and self.input_type == "mnist_flatten":
                image_filename = self.image_paths[index]

                with Image.open(image_filename) as image:
                    image_tensor = MNIST_Flatten_Transform(image)
                return image_tensor.flatten(), self.labels[index]

            image_filename = self.image_paths[index]
            with Image.open(image_filename) as image:
                # for the original images, make sure to convert to RGB
                if self.input_type == "original":
                    image = image.convert("RGB")
            image_tensor = self.pipeline(image)
            return image_tensor, self.labels[index]


if __name__ == "__main__":

    import pandas as pd
    # df = pd.read_csv(f'dataset_key.csv')
    # task_name_list = list(df["new_dataset_name"])
    # with open("full_dataset_stat.csv", "w") as f:
    #     f.write("task_id,task_name,num_classes,train_size,val_size,test_size\n")
    #     for i, task_name in enumerate(task_name_list):
    #         print(f"-------------------------start loading {task_name}: {i}/102-------------------------")
    #         train_dataset = CollectionDataset(task_name, 'train', 'features', vector_type='xception', pipeline=xception_tf_pipeline)
    #         val_dataset = CollectionDataset(task_name, 'validation', 'features', label_dict=train_dataset.label_dict, vector_type='xception', pipeline=xception_tf_pipeline)
    #         test_dataset = CollectionDataset(task_name, 'test', 'features', label_dict=train_dataset.label_dict, vector_type='xception', pipeline=xception_tf_pipeline)
    #         os.makedirs(f"dataset_detail/{task_name}", exist_ok=True)
    #         with open(f"dataset_detail/{task_name}/label_dict.pickle", 'wb') as handle:
    #             pickle.dump(train_dataset.label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #         with open(f"dataset_detail/{task_name}/train.csv", "w") as sf:
    #             sf.write("image_path,image_label,image_hash\n")
    #             for specific_image in train_dataset.images:
    #                 sf.write(f"{specific_image.relative_path},{specific_image.class_id},{specific_image.file_hash}\n")
    #         with open(f"dataset_detail/{task_name}/val.csv", "w") as sf:
    #             sf.write("image_path,image_label,image_hash\n")
    #             for specific_image in val_dataset.images:
    #                 sf.write(f"{specific_image.relative_path},{specific_image.class_id},{specific_image.file_hash}\n")
    #         with open(f"dataset_detail/{task_name}/test.csv", "w") as sf:
    #             sf.write("image_path,image_label,image_hash\n")
    #             for specific_image in test_dataset.images:
    #                 sf.write(f"{specific_image.relative_path},{specific_image.class_id},{specific_image.file_hash}\n")

    #         f.write(f"{i},{task_name},{len(train_dataset.label_dict.keys())},{len(train_dataset)},{len(val_dataset)},{len(test_dataset)}\n")

    with open("cifar100.csv", "w") as f:
        f.write("task_name,task_id,num_classes,train_size,test_size\n")
        for i in range(20):
            train_dataset = CustomDataset("cifar100", i, 'train', 'original')
            test_dataset = CustomDataset("cifar100", i, 'test', 'original')

            f.write(f"{train_dataset.dataset_name},{i},{len(train_dataset.label_dict.keys())},{len(train_dataset)},{len(test_dataset)}\n")