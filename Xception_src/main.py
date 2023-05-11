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

# sys.path.append("/lab/tmpig8d/u/yuecheng/yuecheng_code/")
sys.path.append("../")
from dataset.dataloader_reader import load_dataloader
from dataset.loader import resnet50_tf_pipeline, xception_tf_pipeline




args = sys.argv[1].split(":")
device = torch.device('cuda:'+args[0])

arg_index = int(args[1])
model_type = args[2]
batch_size = int(args[3])
dataset = "decathlon"

log_path = f"/lab/tmpig8d/u/yuecheng/yuecheng_log/SHELL_{dataset}/"
weight_path = f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SHELL_{dataset}"

def log(filename, message, model_type, write_time=False):
    import os
    os.makedirs(log_path, exist_ok=True)
    with open(log_path+filename+f"_{model_type}_{batch_size}.txt", "a") as f:
        if write_time:
            f.write(str(datetime.datetime.now()))
            f.write("\n")
        f.write(str(message))
        f.write("\n")

def train(epoch, task_num, train_loader, model, optimizer, multiple_model=False):
    # if epoch % 10 == 0:
        # log(str(arg_index), f"**********Start Training task:{task_num}, epoch:{epoch}**********", model_type)
    model.eval()
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(train_loader):
        im, target = batch
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
        
    # if epoch % 10 == 0:
        # log(str(arg_index),'Train Epoch: {} Acc_Train: {:.6f}'.format(epoch, float(correct.item()/total)), model_type)

def eval(model, val_loaders, multiple_model=False):
    correct = 0
    total = 0
    class_correct = 0

    if multiple_model:
        class_correct = 0
        for batch_idx, batch in enumerate(val_loaders):
            im, target = batch
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
            # log(str(arg_index),f"{i}th class correction rate is :{class_correct/len(val_loader.dataset)}", model_type)
            class_correct = 0
            total += len(val_loader.dataset)
        # log(str(arg_index),correct/total, model_type)

# def read_file(filename):
#     container = []
#     with open(filename, "r") as f:
#         for line in f.readlines():
#             data = line.strip().split()
#             container.append((data[0],int(data[1])))
#     return container
def add_parameters(params, model, layer_type):
    for name, layer in model.named_children():
        if isinstance(layer, layer_type):
            params += list(layer.parameters())
        params = add_parameters(params, layer, layer_type)
    return params
if __name__ == "__main__":
    # if arg_index == -1:
    #     train_datasets, test_datasets, train_loaders, test_loaders = load_dataloader(arg_index, batch_size=batch_size, pipeline=xception_tf_pipeline, full_dataset=False)
        
        # for task_idx in range(len(train_datasets)):
    train_dataset, test_dataset, train_loader, test_loader = load_dataloader(arg_index, batch_size=batch_size, input_type="original", dataset_name=dataset)
    task_name = train_dataset.dataset_name
    train_num_class = train_dataset.num_classes

    log(str(arg_index), f"start triaining task{arg_index}: {arg_index}", model_type)

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
        # model = Resnet_TB(train_num_class)

    
    if model_type == "Finetune" or model_type == "Scratch":
        params = list(model.parameters())
    else:
        params = list(model.fc.parameters())
    if model_type == "BPN":
        params = add_parameters(params, model, ConvBias_layer)

    # log(str(arg_index),len(params), model_type)

    optimizer = optim.Adam(params)

    model.to(device)
    model.eval()
    best_eval = 0
    best_state_dict = None
    early_stop_index = 1
    for epoch in range(1, 100+1):
        if model_type == "Finetune" or model_type == "Scratch":
            model.train()
        train(epoch, arg_index, train_loader, model, optimizer, True)
        model.eval()
        acc = eval(model, test_loader, True)
        if acc > best_eval:
            best_eval = acc
            best_state_dict = deepcopy(model.state_dict())
            log(str(arg_index),f"epoch:{epoch} accuracy is {acc}", model_type)
            early_stop_index = epoch
        elif epoch - early_stop_index > 10:
            log(str(arg_index),f"more than 10 epochs without change in acc, stop here, epoch:{epoch}", model_type)
            break
        # with open("timing_log.txt", "a") as f:
        #     f.write(f"{datetime.now()}: finish training task{arg_index}, epoch:{epoch}\n")
    
    log(str(arg_index),f"final evaluation accuracy is {best_eval}, current acc is {acc}", model_type)


    with open(f"{log_path}/{model_type}_total.txt", "a") as f:
        f.write(f"{arg_index},{task_name},{best_eval}\n")

    import os
    os.makedirs(f"{weight_path}/", exist_ok=True)
    torch.save(best_state_dict, f"{weight_path}/{model_type}_{task_name}.pth")
# #####################################################################################
#             from thop import profile
#             import timm
            
#             from pypapi import events, papi_high as high
#             for i in (total_classes):
#                 model = timm.create_model('xception',pretrained=True)
#                 for param in model.parameters():
#                     param.requires_grad = False
#                 model.fc = torch.nn.Linear(in_features=2048, out_features=i, bias=True)
#                 input = torch.randn(1, 3, 224, 224)
#                 high.start_counters([events.PAPI_SP_OPS,])
#                 # output = model(input)
#                 # pred = output.data.max(1, keepdim=True)[1]
#                 # target = torch.ones(1, dtype=torch.long)
#                 # ce_loss = F.cross_entropy(output, target)
#                 # ce_loss.backward()
#                 # optimizer.step()
#                 macs, params = profile(model, inputs=(input, ), verbose=False)
#                 print(macs)
#                 fl = high.stop_counters()
#                 # print(fl)

#             import torchvision.models as models
#             import torch
#             from ptflops import get_model_complexity_info
#             from thop import profile

#             with torch.cuda.device(0):
#                 import timm
#                 model = timm.create_model('xception')
#                 # model = models.resnet18()
#                 model.fc = nn.Linear(2048, 38)
#                 macs, params = get_model_complexity_info(model, (3, 299, 299), as_strings=True,
#                                                         print_per_layer_stat=True, flops_units="Mac", verbose=True)
#                 print(macs, params)

#                 input = torch.randn(1, 3, 299, 299)
#                 macs, params = profile(model, inputs=(input, ), verbose=False)
#                 print(macs, params)
            
#             # from torchsummary import summary
#             # import timm
#             # model = timm.create_model('xception')
#             # summary(model, (3, 224, 224), depth=7)