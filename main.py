import os
import argparse
from copy import deepcopy

import torch
import torch.optim as optim
import timm

from Xception_src.classifiers import *
from Xception_src.Conv_BP_layer_prototype import *
from dataset.dataloader_reader import load_dataloader
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default='./result/')
    parser.add_argument('--weight', type=str, default='./weight/')
    parser.add_argument('--data', type=str, default='./data/')
    parser.add_argument('--method', type=str, default="Linear_SKILL", help="Linear_SKILL for linear feature dataset with \
                        fc layer. BB_SKILL for imagedataset with BB implementation. If you want to test the fc performance with a \
                        image folder, simply type Linear")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exist(args.result):
        os.mkdir(args.result)
    if not os.path.exist(args.weight):
        os.mkdir(args.weight)
    # You may want to use your own code for dataset, here we provide datasets we presented in the paper
    ##############################################################################################################
    if args.method == "Linear_SKILL":
        train_datasets, test_datasets, train_loaders, test_loaders = load_dataloader(-1, input_type="features")
    else:
        train_datasets, test_datasets, train_loaders, test_loaders = load_dataloader(-1, input_type="original")
    ##############################################################################################################
    for i in range(len(train_datasets)):
        train_dataset = train_datasets[i]
        test_dataset = test_datasets[i]
        train_loader = train_loaders[i]
        test_loader = test_loaders[i]

        task_name = train_dataset.dataset_name # name for your ith dataset, you can set it with your own method
        train_num_class = train_dataset.num_classes # number of classes of your ith dataset, you can set it with your own method

        if args.method == "Linear":
            model = timm.create_model('xception',pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = torch.nn.Linear(in_features=2048, out_features=train_num_class, bias=True)
        elif args.method == "Linear_SKILL":
            model = torch.nn.Linear(in_features=2048, out_features=train_num_class, bias=True)
        elif args.method == "BB_SKILL":
            model = Xception_TB(train_num_class)

        if args.method == "Linear_SKILL":
            params = list(model.parameters())
        else:
            params = list(model.fc.parameters())
        if args.method == "BB_SKILL":
            params = add_parameters(params, model, ConvBias_layer)
        optimizer = optim.Adam(params)

        model.to(device)
        model.eval() # nothing in the backbone should be changed including mean and variance of BatchNorm
        best_eval = 0
        best_state_dict = None
        early_stop_index = 1
        train_accs, train_losses, val_accs = [], [], []
        for epoch in range(1, 100+1):
            train_acc, train_loss = train(epoch, i, train_loader, model, optimizer, True)
            model.eval()
            val_acc = eval(model, test_loader, True)
            if val_acc > best_eval:
                best_eval = val_acc
                best_state_dict = deepcopy(model.state_dict())
                log(args.result, str(i), f"epoch:{epoch} train acc is {train_acc}, val acc is {val_acc}", args.method)
                early_stop_index = epoch
            elif epoch - early_stop_index > 10:
                log(args.result, str(i), f"more than 10 epochs without change in acc, stop here, epoch:{epoch}", args.method)
                break
        
        log(args.result, str(i), f"final evaluation accuracy is {best_eval}, current acc is {val_acc}", args.method)
        with open(os.path.join(args.result, f"{args.method}_total.txt"), "a") as f:
            f.write(f"{i},{task_name},{best_eval}\n")
        # os.makedirs(f"{weight_path}/", exist_ok=True)
        torch.save(best_state_dict, os.path.join(args.weight, f"{args.method}_{task_name}.pth"))