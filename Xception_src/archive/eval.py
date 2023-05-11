import os
from re import A
from utils import get_files_and_labels
from dataset import ShELL_Dataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import sys
from classifiers import *

device = torch.device('cuda:0')
def train(epoch, task_num, train_loader, model, optimizer, multiple_model=False):
    if epoch % 1 == 0:
        print(f"**********Start Training task:{task_num}, epoch:{epoch}**********")
    model.eval()
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(train_loader):
        im, target, _ = batch
        total+= im.shape[0]

        im, target = im.to(device), target.to(device)
        optimizer.zero_grad()
        
        if multiple_model:
            output = model(im)
        else:
            output = model(im,task_num)
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        ce_loss = F.cross_entropy(output, target)
        loss = ce_loss 
        loss.backward()
        optimizer.step()
    # for name, param in model.layer4[0].conv1.combine.named_parameters():
    #     print(name, param)
        
    if epoch % 1 == 0:
        print('Train Epoch: {} Acc_Train: {:.6f}'.format(epoch, float(correct.item()/total)))

def eval(model, val_loaders, multiple_model=False):
    correct = 0
    total = 0
    class_correct = 0

    if multiple_model:
        class_correct = 0
        for batch_idx, batch in enumerate(val_loaders):
            im, target, _ = batch
            im, target = im.to(device), target.to(device)
            output = model(im)
            pred = output.data.max(1, keepdim=True)[1]
            class_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        return class_correct/len(val_loaders.dataset)
    else:
        for i, val_loader in enumerate(val_loaders):
            for batch_idx, batch in enumerate(val_loader):
                im, target, _ = batch
                im, target = im.to(device), target.to(device)
                output = model(im,i)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                class_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            print(f"{i}th class correction rate is :{class_correct/len(val_loader.dataset)}")
            class_correct = 0
            total += len(val_loader.dataset)
        print(correct/total)

if __name__ == "__main__":
    train_datasets = []
    val_datasets = []
    agents_train_loaders = []
    agents_val_loaders = []
    original_class_list = []
    with open("/lab/tmpig20c/u/zmurdock/shell-datasets/shell_list.txt") as f:
        for line in f.readlines():
            original_class_list.append(line.strip().split()[1])
    # print(len(original_class_list))
    class_list = []
    experiment_files_path = "/lab/tmpig20c/u/zmurdock/shell-datasets"
    experiment_data_path = "/lab/tmpig20c/u/zmurdock/shell-datasets"
    task_id_list = list(range(101))
    skip_ids = []
    class_idx = 0
    for class_id in task_id_list:
        
        ## skip datasets that we don't have right now
        if class_id in skip_ids:
            continue
        class_list.append(original_class_list[class_id])
        ## map the task names to 0-N again, since some datasets are not full processed yet and are skipped
        class_number = class_idx
        class_idx += 1
        
        # clear these for each set
        train_files = []
        train_labels = []
        val_files = []
        val_labels = []
        label_dict = {}
                    
        train_file_name = os.path.join(experiment_files_path, original_class_list[class_id], f"shell_train.txt")
        train_files, train_labels, label_dict = get_files_and_labels(train_file_name, experiment_data_path, train_files, train_labels, label_dict, class_number, False)
        
        val_file_name = os.path.join(experiment_files_path, original_class_list[class_id], f"shell_test.txt")
        val_files, val_labels, _ = get_files_and_labels(val_file_name, experiment_data_path, val_files, val_labels, label_dict, class_number, True)
        train_datasets.append(ShELL_Dataset(class_number, train_files, train_labels, label_dict))
        val_datasets.append(ShELL_Dataset(class_number, val_files, val_labels, label_dict))

    
    agents_train_loaders = [DataLoader(train_datasets[t], batch_size=16,
                                                    shuffle=True) for t in range(len(train_datasets))]
    agents_val_loaders = [DataLoader(val_datasets[t], batch_size=4,
                                                    shuffle=True) for t in range(len(val_datasets))]
    total_classes = [len(train_dataset.labels) for train_dataset in train_datasets]
    
    arg_index = int(sys.argv[1])
    num_class = total_classes[arg_index]

    print(f"start testing tast{arg_index}: {class_list[arg_index]}")
    print(f"test_size is {len(agents_val_loaders[arg_index].dataset)}")
    print(f"test_class num is {len(agents_val_loaders[arg_index].dataset.labels)}")
    # import timm
    # model = timm.create_model('xception',pretrained=True)
    # model.fc = torch.nn.Linear(in_features=2048, out_features=num_class, bias=True)
    # state_dict = torch.load(f"weight/full_dataset/Xception_{class_list[arg_index]}.pth")
    # with torch.no_grad():
    #     model.fc.weight.copy_(state_dict['fc.weight'])
    #     model.fc.bias.copy_(state_dict["fc.bias"])
    # model.to(device)
    
    model = Xception_TB(num_class)
    original_state_dict = model.state_dict()
    state_dict = torch.load(f"weight/full_dataset/Xception_TB_{class_list[arg_index]}.pth")
    print("successfully load state_dict")
    for param in original_state_dict:
        if param[-6:] == "1.bias" or param == "fc.weight" or param == "fc.bias":
            original_state_dict[param] = state_dict[param]
    model.load_state_dict(original_state_dict)
    model.to(device)
    model.eval()
    acc = eval(model, agents_val_loaders[arg_index], True)
    print(val_datasets[arg_index].image_labels[0])
    print(f"final evaluation accuracy for class{arg_index}:({class_list[arg_index]}) is {acc}")