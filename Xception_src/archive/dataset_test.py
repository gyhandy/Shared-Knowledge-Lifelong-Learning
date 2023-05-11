import torch
from torch.utils.data import Dataset
from torchvision import transforms


import os
from PIL import Image

class ShELL_Dataset(Dataset):
    def __init__(self, id, files, file_labels, label_dict, is_cuda=True) -> None:
        self.id = id
        self.label_dict = label_dict
        self.image_labels = []
        for i in range(len(files)):
            self.image_labels.append((files[i], file_labels[i]))
        
        self.transform = transforms.Compose([
                            transforms.Resize(400),
                            transforms.CenterCrop(299),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.is_cuda = is_cuda

    @property
    def labels(self) -> list:
        return list(set([p[1] for p in self.image_labels]))
        #return list(self.label_dict.values()) ## label_dict seems unnecessary

    def __len__(self) -> int:
        return len(self.image_labels)

    def __getitem__(self, idx):
        try:
            img_path, class_label = self.image_labels[idx]
            image: Image.Image = Image.open(img_path)
            image = image.convert("RGB")
            
            image_tensor: torch.Tensor = self.transform(image)
        
            return image_tensor, class_label, self.id
        except FileNotFoundError as e:
            with open(f"/lab/harry/100_full_dataset_error/task{self.id}_error.txt", "a") as f:
                f.write(str(e))
                f.write("\n")
            return torch.rand((3, 299, 299)), 1, 1

        except OSError as ose:
            img_path, class_label = self.image_labels[idx]
            
            with open(f"/lab/harry/100_full_dataset_error/task{self.id}_error.txt", "a") as f:
                f.write(str(ose))
                f.write(": " + img_path)
                f.write("\n")
            return torch.rand((3, 299, 299)), 1, 1