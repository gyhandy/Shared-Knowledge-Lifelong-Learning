import torch
import torch.nn.functional as F

from datetime import datetime

import yaml
import os
import numpy as np
import random
import shutil
import json
from torch.nn import Softmax

# following is the helper function for BB
def log(log_path, filename, message, model_type, write_time=False):
    import os
    os.makedirs(log_path, exist_ok=True)
    with open(log_path+filename+f"_{model_type}.txt", "a") as f:
        if write_time:
            f.write(str(datetime.now()))
            f.write("\n")
        f.write(str(message))
        f.write("\n")

def train(train_loader, model, optimizer, device=torch.device("cuda:0")):
    model.eval() # backbone shouldn't learn new mean and variance since they are all fixed
    correct = 0
    total = 0
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
        loss.backward()
        optimizer.step()

    return correct/total, loss

def eval(model, val_loaders, device=torch.device("cuda:0")):
    model.eval()
    correct = 0
    total = 0
    class_correct = 0

    class_correct = 0
    for batch_idx, batch in enumerate(val_loaders):
        im, target = batch
        im, target = im.to(device), target.to(device)
        output = model(im)
        pred = output.data.max(1, keepdim=True)[1]
        class_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    return class_correct/len(val_loaders.dataset)

def add_parameters(params, model, layer_type):
    for name, layer in model.named_children():
        if isinstance(layer, layer_type):
            params += list(layer.parameters())
        params = add_parameters(params, layer, layer_type)
    return params




# following is the helper function for gmmc
def read_file(filename):
    container = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            container.append((data[0],int(data[1])))
    return container

def setup(args):
    makedirectory(args['dir_results'])
    if args['test_num']>=0:
        save_name = 'Test_%d'%(args['test_num'])
    else:
        save_name = datetime.now().strftime(r'%d%m%y_%H%M%S')
    dir_save = '%s/%s/%s/'%(args['dir_results'], args['dset_name'], save_name)
    # check if exists, if yes, overwrite. 
    if os.path.exists(dir_save) and os.path.isdir(dir_save):
        shutil.rmtree(dir_save)
    makedirectory(dir_save)
    # save config of experiment
    #dict_args = vars(args)
    with open('%sconfig_model.json'%(dir_save), 'w') as fp:
        json.dump(args, fp, sort_keys=True, indent=1)
        
        
    args['acc_tmapping_student'] = '%s/%s.txt'%(dir_save, 'student_task_mapping_acc')
    args['acc_tmapping_student_per_task'] = '%s/%s.txt'%(dir_save, 'student_task_mapping_acc_per_task')
    args['acc_teacher'] = '%s/%s.txt'%(dir_save, 'teacher_acc')
    args['acc_student'] = '%s/%s.txt'%(dir_save, 'student_acc')

    args['dir_save'] = dir_save
    
    return args

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def makedirectory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def divide_integer_K(N,K, shuff=True):
    '''Divide an integer into equal parts exactly'''
    array=np.zeros(K,)
    for i in range(K):
        array[i] = int(N / K)    # integer division
    # divide up the remainder
    for i in range(N%K):
        array[i] += 1
    array = array.astype(int)
    if shuff:
        np.random.shuffle(array)
    return array

def get_args_from_yaml(args, params_parser):
    with open(args['config_file']) as fid:
        args_yaml = yaml.load(fid, Loader=yaml.SafeLoader)
    ad = args.__dict__
    # print(args_yaml)
    for k in ad.keys():
        dv = params_parser.get_default(k)
        if dv is not None:  # ad[k] will not be None if it has a default value
            if ad[k] == dv and k in args_yaml:
                ad[k] = args_yaml[k]
        elif ad[k] is None:
            if k in args_yaml:
                ad[k] = args_yaml[k]
    return args

def get_files_and_labels(dataset_file_name: str, base_file_path: str, files: list, labels: list, label_dict, dset_idx, flag_unknown_label:bool=False):
    
    # check if label_dict already contains entries
    if len(label_dict.values()) == 0:
        next_label_to_use = 0 
    else:
        next_label_to_use = list(label_dict.values())[-1] + 1

    with open(dataset_file_name) as f:
        lines = f.readlines()
        for line in lines:
            
            ## Some file paths have spaces in them which breaks the split by space. (scenes have 4 cases like this, wikiart 14)
            if len(line.strip().split()) > 2: 
                #print(line)
                continue
                ## If there's a file name with space between words, this puts it together
                #individual_file = " ".join(line.strip().split()[0:-1])
                #individual_label = line.strip().split()[-1]
            try:
                individual_file, individual_label = line.strip().split() ## All of them were written to be split by a space
            
                individual_label = (dset_idx, int(individual_label))
                if individual_label not in label_dict:

                    if flag_unknown_label:
                        #breakpoint()
                        # print(line)
                        continue
                        #raise ValueError
                    label_dict[individual_label] = next_label_to_use
                    next_label_to_use += 1
                
                files.append(os.path.join(base_file_path, individual_file))
                labels.append(label_dict[individual_label])
            except:
                print(dataset_file_name)
    return files, labels, label_dict

def inference(model, dataloader, device=torch.device("cuda:0")):
    outputs = []
    labels = []
    softmax = Softmax(dim=1)
    for i, data in enumerate(dataloader):
        image = data[0]
        label = data[1]
        image, label = image.to(device), label.to(device)
        output = model(image)
        output = softmax(output)
        output = torch.argmax(output, dim=1)
        outputs.extend(output.tolist())
        labels.extend(label.tolist())
    labels = np.array(labels)
    outputs = np.array(outputs)
    error_rate = np.count_nonzero(labels-outputs)/len(labels)
    # print(f"correction rate: {1-error_rate}")
    return [labels[i] == outputs[i] for i in range(len(labels))]