import torch
import torch.optim as optim
import sys
import pandas as pd
from utils import *
from copy import deepcopy
import numpy as np
from normalized_lc import Normalize_FC as NFC
from datetime import datetime

sys.path.append("/lab/harry/WorkSpace/yuecheng_code")
from dataset.dataloader_reader import load_dataloader

device = torch.device('cuda:0')
mode = "normal"
dataset_name = "decathlon"
log_folder = f"/lab/harry/WorkSpace/yuecheng_code/weight_combining/result/log/{dataset_name}/{mode}_norm_log/"
weight_folder = f"/lab/harry/WorkSpace/yuecheng_code/weight_combining/result/weight/{dataset_name}/{mode}_norm_weight/"
epochs=60
step_size=20
lr=1e-3
gamma=0.1
start = 1
end = 10
# start = int(sys.argv[1])
# end = int(sys.argv[2])

if __name__ == "__main__":
    check_path(log_folder)
    check_path(weight_folder)
    check_path(f"result/summary_result/{dataset_name}")
    id_name_dict = {}
    id_num_classes_dict = {}
    id_accs = {}
    
    
    for i in range(start, end):
        # train_dataset, val_dataset, train_loader, test_loader = load_dataloader(i, 32, input_type="features", vector_type="xception")
        train_dataset, val_dataset, train_loader, test_loader = load_dataloader(i, 32, input_type="features", dataset_name="decathlon")
        id_name_dict[i] = train_dataset.dataset_name
        id_num_classes_dict[i] = train_dataset.num_classes
        """Here to make sure if dataset make sense"""
        with open(log_folder+f"{i}_{id_name_dict[i]}_dataset_peek.txt", "w") as f:
            f.write(f"dataset_name {id_name_dict[i]}\n")
            f.write(f"number of classes: {train_dataset.num_classes}\n")
            f.write(f"label_dictionary: {train_dataset.label_dict}\n")
        if mode != "normal":
            linear_classifier = NFC(id_num_classes_dict[i], setting_bias=True, norm_usage=mode).to(device)
        else:
            linear_classifier = torch.nn.Linear(2048, id_num_classes_dict[i]).to(device)
        optimizer = optim.Adam(linear_classifier.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        accs = []
        best_eval = 0
        best_state_dict = None
        for epoch in range(1, epochs+1):
            log(str(i)+"_"+id_name_dict[i]+"_training_log", f"**********Start Training task:{id_name_dict[i]}, epoch:{epoch}**********", log_folder, True)
            loss, train_acc = train(device, train_loader, linear_classifier, optimizer, scheduler=scheduler)
            log(str(i)+"_"+id_name_dict[i]+"_training_log", f'Train Epoch: {epoch} Train Loss: {loss} Acc_Train: {train_acc}', log_folder)
            acc = eval(linear_classifier, test_loader, device)
            
            accs.append(acc)
            if acc > best_eval:
                best_eval = acc
                best_state_dict = deepcopy(linear_classifier.state_dict())
                log(str(i)+"_"+id_name_dict[i]+"_training_log", f"epoch:{epoch} accuracy is {acc}", log_folder)
        log(str(i)+"_"+id_name_dict[i]+"_training_log", f"accs are {accs}", log_folder)
        log(str(i)+"_"+id_name_dict[i]+"_training_log", f"best acc is {best_eval} or {max(accs)}", log_folder)
        id_accs[i] = best_eval
        torch.save(best_state_dict, weight_folder+f"{i}_{id_name_dict[i]}.pth")
        linear_classifier.load_state_dict(best_state_dict)

        """
        we want to know more about classifier now
        """
        with open(log_folder+f"{i}_{id_name_dict[i]}_final_result_log.txt", "w") as f:
            for batch_idx, batch in enumerate(test_loader):
                im, target = batch
                im = im.to(device)
                output = linear_classifier(im)
                pred = output.data.max(1, keepdim=True)[1]
                for result_idx in range(len(output)):
                    single_result = output[result_idx]
                    single_result = single_result.detach().cpu().numpy()
                    single_pred = pred[result_idx].detach().cpu().numpy()
                    f.write(f"{np.sort(single_result)[-1:-6:-1]} {np.argsort(single_result)[-1:-6:-1]} {single_pred} {target[result_idx]}\n")
    with open(f"result/summary_result/{dataset_name}/{mode}_norm_training.csv", "a") as f:
        for index in range(start, end):
            f.write(f"{index},{id_name_dict[index]},{id_accs[index]}\n")
    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)