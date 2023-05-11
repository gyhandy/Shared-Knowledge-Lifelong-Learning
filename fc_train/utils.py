import torch
import torch.nn.functional as F
import os
from datetime import datetime
import time

def log(filename, message, log_folder, write_time=False):
    with open(log_folder+filename+".txt", "a") as f:
        if write_time:
            f.write(str(datetime.now()))
            f.write("\n")
        f.write(str(message))
        f.write("\n")

def check_path(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except Exception as e:
        print("bug may occur when multiple processes or multiple machines are used")

def train(device, train_loader, model, optimizer, scheduler=None, reg="l2", lamda=1e-5):
    correct = 0
    total = 0
    total_loss = 0
    total_time = 0
    for batch_idx, batch in enumerate(train_loader):
        im, target = batch
        total+= im.shape[0]

        im, target = im.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(im)
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        ce_loss = F.cross_entropy(output, target)
        loss = ce_loss

        total_loss += loss
        loss.backward()
        optimizer.step()
    if scheduler != None:
        scheduler.step()
    
    total_loss = float(total_loss)
    accuracy = float(correct.item()/total)
    return total_loss, accuracy

def eval(model, val_loaders, device):
    class_correct = 0
    for batch_idx, batch in enumerate(val_loaders):
        im, target = batch
        im, target = im.to(device), target.to(device)
        output = model(im)
        pred = output.data.max(1, keepdim=True)[1]
        class_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    return class_correct/len(val_loaders.dataset)

def weight_normalize(loaded_weight, mode):
    """
    -input
        -loaded_weight unnormalized weight directly load from .pth
        -norm type originally used
    """
    weight = loaded_weight["fc.weight"]
    bias = loaded_weight["fc.bias"]
    concat_matrix = torch.cat([weight, bias.unsqueeze(1)], dim=1)
    if mode == 0:
        norm, _ = torch.max(torch.abs(concat_matrix), dim=1)
    else:
        norm = concat_matrix.norm(p=mode, dim=1)
    concat_matrix = concat_matrix.transpose(0, 1)
    concat_matrix = (concat_matrix/norm).transpose(0, 1)
    return concat_matrix[:, :2048], concat_matrix[:,-1].flatten()

def weight_combining(weights, bias, index=None):
    """
    -input
        -weights list of weights each has n*2048 which is the weight of the task
        -bias list of bias each has n, which is the bias of the task
        -index list of tuple, (task, class) task id and class id (task id is the index of the task in the combining process)
    """
    task_length = [len(bia) for bia in bias]
    task_length_sum = [sum(task_length[:i]) for i in range(len(task_length))]
    print(task_length_sum)
    if index == None:
        result_weight = torch.cat(weights, dim=0)
        result_bias = torch.cat(bias, dim=0)
    else:
        new_index = [task_length_sum[a]+b for a, b in index]
        result_weight = torch.cat(weights, dim=0)[new_index]
        result_bias = torch.cat(bias, dim=0)[new_index]
    if index==None:
        fc = torch.nn.Linear(2048, sum(task_length))
    else:
        fc = torch.nn.Linear(2048, len(index))
    with torch.no_grad():
        fc.weight.copy_(result_weight)
        fc.bias.copy_(result_bias)
    return fc

if __name__ == "__main__":
    mode = 0
    device = torch.device('cuda:0')
    lc1_weight = torch.load(f"/lab/harry/WorkSpace/yuecheng_code/weight_combining/weight/{mode}_norm_weight/1_MIT_Indoor_Scenes.pth", map_location=device)
    lc4_weight = torch.load(f"/lab/harry/WorkSpace/yuecheng_code/weight_combining/weight/{mode}_norm_weight/4_Fine-Grained_Visual_Classification_of_Aircraft.pth", map_location=device)
    lc1_w, lc1_b = weight_normalize(lc1_weight, mode)
    lc4_w, lc4_b = weight_normalize(lc4_weight, mode)
    fc = weight_combining([lc1_w, lc4_w], [lc1_b, lc4_b])


