from dataset.loader import CollectionDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
from dataset.custom_loader import CustomDataset
from dataset.decathlon_loader import Decathlon_Dataset
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
        dataset_name = "SKILL"):
    print("  ||||")
    print("  ||||")
    print("  ||||")
    print("  ||||")
    print("  ||||")
    print("\\      /")
    print(" \\    /")
    print("  \\  /")
    print("   \\/")
    print("----------------------------------------------------------------------------------------------------------------------------------------------")
    print("Please read:")
    print("if you are using decathlon: Please make sure you are using full_dataset=True unless you know exactly the reason and logic of this argument!!!")
    print("----------------------------------------------------------------------------------------------------------------------------------------------")
    print("   /\\")
    print("  /  \\")
    print(" /    \\")
    print("/      \\")
    print("  ||||")
    print("  ||||")
    print("  ||||")
    print("  ||||")
    print("  ||||")

    current_path = os.path.dirname(os.path.abspath(__file__))
    if dataset_name == "SKILL":
        
        df = pd.read_csv(f'{current_path}/full_dataset_stat.csv')
        task_name_list = list(df["task_name"])

        if index == -1:
            train_datasets = []
            test_datasets = []
            for task_name in task_name_list:
                train_dataset = CollectionDataset(task_name, 'train', input_type=input_type, vector_type=vector_type, pipeline=pipeline, full_dataset=full_dataset)
                test_dataset = CollectionDataset(task_name, 'test', input_type=input_type, vector_type=vector_type, pipeline=pipeline, label_dict=train_dataset.label_dict, full_dataset=full_dataset)
                train_datasets.append(train_dataset)
                test_datasets.append(test_dataset)
            train_loaders = [DataLoader(td, batch_size=batch_size, shuffle=shuffle_train) for td in train_datasets]
            test_batch_size = batch_size//2
            if test_batch_size == 0:
                test_batch_size = 1
            test_loaders = [DataLoader(td, batch_size=test_batch_size, shuffle=shuffle_test) for td in test_datasets]
            return train_datasets, test_datasets, train_loaders, test_loaders 
        else:
            train_dataset =  CollectionDataset(task_name_list[index], 'train', input_type=input_type, vector_type=vector_type, pipeline=pipeline, full_dataset=full_dataset)
            test_dataset = CollectionDataset(task_name_list[index], 'test', input_type=input_type, vector_type=vector_type, pipeline=pipeline, label_dict=train_dataset.label_dict, full_dataset=full_dataset)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        test_batch_size = batch_size//2
        if test_batch_size == 0:
            test_batch_size = 1
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle_test)
        return train_dataset, test_dataset, train_loader, test_loader
    elif dataset_name == "mnist" or dataset_name == "cifar100":
        if dataset_name == "mnist":
            task_num = 10
        elif dataset_name == "cifar100":
            task_num = 20
        if index == -1:
            train_datasets = []
            test_datasets = []
            for i in range(task_num):
                train_dataset = CustomDataset(dataset_name, i, 'train', input_type=input_type)
                test_dataset = CustomDataset(dataset_name, i, 'test', input_type=input_type)
                train_datasets.append(train_dataset)
                test_datasets.append(test_dataset)
            train_loaders = [DataLoader(td, batch_size=batch_size, shuffle=shuffle_train) for td in train_datasets]
            test_batch_size = batch_size//2
            if test_batch_size == 0:
                test_batch_size = 1
            test_loaders = [DataLoader(td, batch_size=test_batch_size, shuffle=shuffle_test) for td in test_datasets]
            return train_datasets, test_datasets, train_loaders, test_loaders 
        else:
            assert index < task_num
            train_dataset = CustomDataset(dataset_name, index, 'train', input_type=input_type)
            test_dataset = CustomDataset(dataset_name, index, 'test', input_type=input_type)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
            test_batch_size = batch_size//2
            if test_batch_size == 0:
                test_batch_size = 1
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle_test)
            return train_dataset, test_dataset, train_loader, test_loader
    elif dataset_name == "decathlon":
        task_num = 10
        if index == -1:
            train_datasets = []
            test_datasets = []
            for i in range(task_num):
                if not full_dataset and i == 0 and input_type == "features":
                    train_dataset = Decathlon_Dataset(i, 'train', input_type=input_type, full_dataset=full_dataset)
                else:
                    train_dataset = Decathlon_Dataset(i, 'train', input_type=input_type)
                test_dataset = Decathlon_Dataset(i, 'val', input_type=input_type)
                train_datasets.append(train_dataset)
                test_datasets.append(test_dataset)
            train_loaders = [DataLoader(td, batch_size=batch_size, shuffle=shuffle_train) for td in train_datasets]
            test_batch_size = batch_size//2
            if test_batch_size == 0:
                test_batch_size = 1
            test_loaders = [DataLoader(td, batch_size=test_batch_size, shuffle=shuffle_test) for td in test_datasets]
            return train_datasets, test_datasets, train_loaders, test_loaders 
        else:
            assert index < task_num
            if not full_dataset and index == 0 and input_type == "features":
                train_dataset = Decathlon_Dataset(index, 'train', input_type=input_type, full_dataset=full_dataset)
            else:
                train_dataset = Decathlon_Dataset(index, 'train', input_type=input_type)
            test_dataset = Decathlon_Dataset(index, 'val', input_type=input_type)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
            test_batch_size = batch_size//2
            if test_batch_size == 0:
                test_batch_size = 1
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle_test)
            return train_dataset, test_dataset, train_loader, test_loader


if __name__ == "__main__":
    tr, te, _, _ = load_dataloader(-1, dataset_name="decathlon")

    with open("decathlon_stat.csv", "w") as f:
        f.write("task_id,task_name,num_classes,train_size,test_size\n")
        for i in range(len(tr)):
        
            
            task_name = tr[i].dataset_name
            num_classes = tr[i].num_classes
            train_size = len(tr[i])
            test_size = len(te[i])
            f.write(f"{i},{task_name},{num_classes},{train_size},{test_size}\n")
