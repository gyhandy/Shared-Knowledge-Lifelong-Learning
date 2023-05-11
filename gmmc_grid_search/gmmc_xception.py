import numpy as np
import yaml
import torch

import utils
from utils import *
import TaskMappers.proto_mapper as pmap
from classifiers import *

import sys
sys.path.append("/lab/tmpig8d/u/yuecheng/yuecheng_code")
sys.path.append("../../")
from dataset.dataloader_reader import *
from datetime import datetime

# setting hyper-parameter
num_task = 10
dataset_name = "decathlon"
batch_size = 32
n_c = 25
device = torch.device("cuda:0")
file_name = f"decathlon{n_c}"
log(file_name, f"using gpux")
def check_path(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except Exception as e:
        print("bug may occur when multiple processes or multiple machines are used")
# creating datasets and dataloadres
import torchvision.transforms as TF
from torchvision.transforms.functional import InterpolationMode
# xception_tf_pipeline = TF.Compose(
#     [
#         TF.Resize(
#             size=333,
#             interpolation=InterpolationMode.BICUBIC,
#             max_size=None,
#             antialias=None,
#         ),
#         TF.CenterCrop(size=(299, 299)),
#         TF.ToTensor(),
#         TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ]
#     )

origin_train_datasets, origin_val_dataset, origin_agents_train_loaders, origin_agents_val_loaders = load_dataloader(-1, 32, shuffle_test=False, dataset_name=dataset_name, full_dataset=True)
feature_train_datasets, feature_val_datasets, feature_train_loaders, feature_val_loaders = load_dataloader(-1, 32, shuffle_test=False, input_type="features", dataset_name=dataset_name, full_dataset=True)
imagnet_train_datasets, _, imagnet_train_loader, _ = load_dataloader(0, 32, shuffle_test=False, input_type="features", dataset_name=dataset_name, full_dataset=False)

total_classes = [train_dataset.num_classes for train_dataset in origin_train_datasets]
class_list = [train_dataset.dataset_name for train_dataset in origin_train_datasets]

# # generate Xcepton results, only need once
# for i in range(len(class_list)):
#     print("doint i")
#     fc = torch.nn.Linear(in_features=2048, out_features=total_classes[i], bias=True)
#     state_dict = torch.load(f"/lab/harry/WorkSpace/yuecheng_code/weight_combining/result/weight/{dataset_name}/normal_norm_weight/{i}_{class_list[i]}.pth", map_location=device)
#     with torch.no_grad():
#         fc.weight.copy_(state_dict['weight'])
#         fc.bias.copy_(state_dict["bias"])
#     fc.to(device)
#     fc.eval()

#     xception_result = inference(fc, feature_val_loaders[i])
#     check_path(f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SKILL_gmmc/{dataset_name}/")
#     torch.save(xception_result, f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SKILL_gmmc/{dataset_name}/Linear_{class_list[i]}_result.pth")

#     model = Xception_TB(total_classes[i])
#     original_state_dict = model.state_dict()
#     # st = torch.load(f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SKILL_final/{model_type}_{task_name}.pth")
#     # model.load_state_dict(st)
#     state_dict = torch.load(f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SHELL_{dataset_name}/BPN_{class_list[i]}.pth", map_location=device)
#     for param in original_state_dict:
#         if param[-6:] == "1.bias" or param == "fc.weight" or param == "fc.bias":
#             original_state_dict[param] = state_dict[param]
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()

#     BPN_result = inference(model, origin_agents_val_loaders[i])
#     log(file_name, f"BPN model accuracy for task: {class_list[i]} is {sum(BPN_result)}")
#     check_path(f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SKILL_gmmc/{dataset_name}/")
#     torch.save(BPN_result, f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SKILL_gmmc/{dataset_name}/BPN_{class_list[i]}_result.pth")

# load result of Xception and Xception BPN
xception_results =[]
tb_results = []
for i in range(len(class_list)):
    xception_result = torch.load(f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SKILL_gmmc/{dataset_name}/Linear_{class_list[i]}_result.pth")
    xception_results.append(xception_result)

    tb_result = torch.load(f"/lab/tmpig8d/u/yuecheng/yuecheng_weight/SKILL_gmmc/{dataset_name}/BPN_{class_list[i]}_result.pth")
    tb_results.append(tb_result)
    # log(file_name, f"finish evaluating {class_list[i]}, results are {sum(xception_result)/len(xception_result)}, {sum(tb_result)/len(tb_result)}")

# setting for gmmc
GMMC_Xception_accs = []
GMMC_TB_xception_accs = []
with open("Shell_8dset.yaml", 'r') as stream:
    try:
        args=yaml.safe_load(stream)
        # log(file_name, str(args), write_time=True)
    except yaml.YAMLError as exc:
        log(file_name, str(exc), write_time=True)
args = setup(args)
utils.seed_torch(args['seed'])
torch.set_num_threads(args['num_threads'])
pin_memory=False
tmapper = pmap.ProtoTaskMapper(args['task_mapper_params'])

def write_result(filename, result):
    with open(filename, "a") as f:
        f.write(str(result)[1:-1])
        f.write("\n")

def test_task_mapping(currentTask, task_mapper, record_detail=False):
    gmmc = [0 for _ in range(num_task)]
    xce = [0 for _ in range(num_task)]
    tb = [0 for _ in range(num_task)]
    for task in range(currentTask+1):
        t = task
        num_samples = len(feature_val_datasets[t])
        targets_task = [t] * num_samples
        preds_task = []
        for val_feat, _ in feature_val_datasets[t]:
            preds_task.append(int(task_mapper.predict_task(val_feat)))
        GMMC_result = [targets_task[i]==preds_task[i] for i in range(len(preds_task))]
        if record_detail:
            with open(f"/lab/tmpig8d/u/yuecheng/yuecheng_log/skill_gmmc_result/{class_list[t]}.txt", "w") as f:
                for k in range(num_samples):
                    f.write(f"{origin_val_dataset[t].images[k].relative_path}, {targets_task[k]}, {preds_task[k]}\n")

        Xception_result = xception_results[task]
        TB_result = tb_results[task]

        task_result = [GMMC_result[i] and Xception_result[i] for i in range(len(GMMC_result))]
        TB_task_result = [GMMC_result[i] and TB_result[i] for i in range(len(GMMC_result))]

        gmmc[t] = (sum(GMMC_result)/len(GMMC_result))
        xce[t] = (sum(task_result)/len(GMMC_result))
        tb[t] = (sum(TB_task_result)/len(GMMC_result))
    return gmmc, xce, tb


gmmc_accs = [[] for _ in range(num_task)]
xce_accs = [[] for _ in range(num_task)]
tb_accs = [[] for _ in range(num_task)]
for t in range(num_task):
    # with open("timing_log.txt", "a") as f:
    #     f.write(f"{datetime.now()}: start training task{t}")
    X = []

    if t == 0:
        for i in range(len(imagnet_train_datasets)):
            X.append(imagnet_train_datasets[i][0].unsqueeze(0))
    else:
        for i in range(len(feature_train_datasets[t])):
            X.append(feature_train_datasets[t][i][0].unsqueeze(0))
    X = torch.cat(X, dim=0).numpy()
    tmapper.fit_task(n_c, t, X)

    gmmc_acc, xce_acc, tb_acc = test_task_mapping(t, tmapper)
    write_result(f"/lab/tmpig8d/u/yuecheng/yuecheng_code/ShELL-classification/gmmc_grid_search/output/gmmc_{dataset_name}_{n_c}.csv", gmmc_acc)
    write_result(f"/lab/tmpig8d/u/yuecheng/yuecheng_code/ShELL-classification/gmmc_grid_search/output/xce_{dataset_name}_{n_c}.csv", xce_acc)
    write_result(f"/lab/tmpig8d/u/yuecheng/yuecheng_code/ShELL-classification/gmmc_grid_search/output/tb_{dataset_name}_{n_c}.csv", tb_acc)
    log(file_name, f"-------------------------------------------------------------------------------------")
    for index in range(len(gmmc_acc)):
        gmmc_accs[index].append(gmmc_acc[index])
        xce_accs[index].append(xce_acc[index])
        tb_accs[index].append(tb_acc[index])
    log(file_name, f"finish training task {t}")
    for index in range(t+1):
        log(file_name, f"task{index} accuracy when {t} total tasks have been trained")
        log(file_name, f"gmmc_accs are {gmmc_accs[index]}")
        log(file_name, f"xce_accs are {xce_accs[index]}")
        log(file_name, f"tb_accs are {tb_accs[index]}")
        log(file_name, f"-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-")
    log(file_name, f"-------------------------------------------------------------------------------------")
# test_task_mapping(num_task-1, tmapper, record_detail=True)