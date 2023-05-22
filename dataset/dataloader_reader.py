from dataset.loader import CollectionDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
import torchvision.transforms as TF
from torchvision.transforms.functional import InterpolationMode


resnet50_tf_pipeline = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

# resize, crop, and normalize (matches pipeline for xception in timm)
xception_tf_pipeline = TF.Compose(
    [
        TF.Resize(
            size=333,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias=None,
        ),
        TF.CenterCrop(size=(299, 299)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

def load_dataloader(index,
        batch_size=32,
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
        input_type="original",
        vector_type="xception",
        pipeline=xception_tf_pipeline,
        full_dataset = False,
        dataset_name = "SKILL",
        root_path = "/lab/tmpig15b/u/"):

    current_path = os.path.dirname(os.path.abspath(__file__))
    if dataset_name == "SKILL":
        
        df = pd.read_csv(f'{current_path}/full_dataset_stat.csv')
        task_name_list = list(df["task_name"])

        if index == -1:
            train_datasets = []
            test_datasets = []
            for task_name in task_name_list:
                train_dataset = CollectionDataset(task_name, 'train', input_type=input_type, vector_type=vector_type, pipeline=pipeline, full_dataset=full_dataset, root_path=root_path)
                test_dataset = CollectionDataset(task_name, 'test', input_type=input_type, vector_type=vector_type, pipeline=pipeline, label_dict=train_dataset.label_dict, full_dataset=full_dataset, root_path=root_path)
                train_datasets.append(train_dataset)
                test_datasets.append(test_dataset)
            train_loaders = [DataLoader(td, batch_size=batch_size, shuffle=shuffle_train) for td in train_datasets]
            test_batch_size = batch_size//2
            if test_batch_size == 0:
                test_batch_size = 1
            test_loaders = [DataLoader(td, batch_size=test_batch_size, shuffle=shuffle_test) for td in test_datasets]
            return train_datasets, test_datasets, train_loaders, test_loaders 
        else:
            train_dataset =  CollectionDataset(task_name_list[index], 'train', input_type=input_type, vector_type=vector_type, pipeline=pipeline, full_dataset=full_dataset, root_path=root_path)
            test_dataset = CollectionDataset(task_name_list[index], 'test', input_type=input_type, vector_type=vector_type, pipeline=pipeline, label_dict=train_dataset.label_dict, full_dataset=full_dataset, root_path=root_path)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        test_batch_size = batch_size//2
        if test_batch_size == 0:
            test_batch_size = 1
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle_test)
        return train_dataset, test_dataset, train_loader, test_loader

if __name__ == "__main__":
    pass