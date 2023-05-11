import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import sys
from classifiers import *
from Conv_BP_layer_prototype import *
from copy import deepcopy
from datetime import datetime
import time
import sys
from tqdm import tqdm

sys.path.append("/lab/tmpig8e/u/yuecheng/yuecheng_code")
from dataset.dataloader_reader import load_dataloader
from dataset.loader import resnet50_tf_pipeline, xception_tf_pipeline


args = sys.argv[1].split(":")
device = torch.device('cuda:'+args[0])

arg_index = int(args[1])
model_type = args[2]
batch_size = int(args[3])

def eval(model, val_loaders, multiple_model=False):
    correct = 0
    total = 0
    class_correct = 0

    if multiple_model:
        class_correct = 0
        for batch_idx, batch in tqdm(enumerate(val_loaders)):
            im, target = batch
            im, target = im.to(device), target.to(device)
            output = model(im)
            pred = output.data.max(1, keepdim=True)[1]
            class_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        return class_correct/len(val_loaders.dataset)
    else:
        for i, val_loader in tqdm(enumerate(val_loaders)):
            for batch_idx, batch in enumerate(val_loader):
                im, target, _ = batch
                im, target = im.to(device), target.to(device)
                output = model(im,i)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                class_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            class_correct = 0
            total += len(val_loader.dataset)

def add_parameters(params, model, layer_type):
    for name, layer in model.named_children():
        if isinstance(layer, layer_type):
            params += list(layer.parameters())
        params = add_parameters(params, layer, layer_type)
    return params

if __name__ == "__main__":
    train_dataset, test_dataset, train_loader, test_loader = load_dataloader(arg_index, batch_size=batch_size, pipeline=xception_tf_pipeline, full_dataset=False)
    task_name = train_dataset.dataset_name
    train_num_class = train_dataset.num_classes

    if model_type == "Linear":
        import timm
        model = timm.create_model('xception',pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(in_features=2048, out_features=train_num_class, bias=True)
    elif model_type == "Finetune":
        import timm
        model = timm.create_model('xception',pretrained=True)
        model.fc = torch.nn.Linear(in_features=2048, out_features=train_num_class, bias=True)
    elif model_type == "Scratch":
        model = resnet18()
        model.fc = torch.nn.Linear(in_features=512, out_features=train_num_class, bias=True)
    elif model_type == "BPN":
        model = Xception_TB(train_num_class)

    model.load_state_dict(torch.load(f"/lab/tmpig8e/u/yuecheng/yuecheng_share/BPN_WEIGHT/BPN_{task_name}.pth"))
    model.to(device)
    model.eval()

    print("finished initialize model")

    acc = eval(model, test_loader, True)
    print(arg_index, task_name, acc)