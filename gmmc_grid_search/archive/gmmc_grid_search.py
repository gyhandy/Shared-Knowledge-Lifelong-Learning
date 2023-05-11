from re import L
import numpy as np
import os
import torch
import yaml
import TaskMappers.proto_mapper_grid as pmap

import utils
from utils import *
from datetime import datetime
import sys
from pypapi import events, papi_high as high
import time

args = sys.argv[1].split(":")

task_num = int(args[1])
cluster_num = int(args[2])

with open("/lab/tmpig8e/u/yuecheng/yuecheng_code/ShELL-classification/gmmc_grid_search/Shell_8dset.yaml", 'r') as stream:
    try:
        args=yaml.safe_load(stream)
        log(f"t{task_num}_clu{cluster_num}", str(args), write_time=True)
    except yaml.YAMLError as exc:
        log(f"t{task_num}_clu{cluster_num}", str(exc), write_time=True)

args = setup(args)
utils.seed_torch(args['seed'])
torch.set_num_threads(args['num_threads'])
pin_memory=False

tmapper = pmap.ProtoTaskMapper(args['task_mapper_params'])

features_path = {'train':[], 'val':[]}
original_class_list = ['flowers', 'scenes', 'birds', 'cars', 'aircraft', 'voc', 'chars', 'svhn', 'Reptilia', 'Fungi', 'Amphibia', 'Arachnida', 'Mollusca', 'Actinopterygii', 'Insecta', 'core50_128', 'Sketches', 'wikiart', 'DescribableTextures', 'GTSRB', 'CelebA', 'OfficeHome_Clipart', 'OfficeHome_Product', 'OfficeHome_Art', 'Food-101', 'EuroSAT', 'CamelyonPatches', 'diabetic-retinopathy-detection', 'RVL-CDIP', 'HistAerial', 'OriSet_classification', 'Brazilian_Coins', 'imaterialist-fashion-2019-FGVC6', 'Rice_Image_Dataset', 'Vegetable_images_Dataset', 'garbage_classification', 'Facial_Expressions_Dataset', 'PokemonData', 'Manga_Facial_Expressions', 'Monkey_Species', 'oregon_wildlife', 'Blood_Cell_Dataset', 'OCT2017', 'aptos2019', 'Cataract_Dataset', 'freiburg_groceries_dataset', 'Fashion_Products_Dataset', 'OnePiece_Dataset', 'Apparel_Images_Dataset', 'Zalando_Fashion_Dataset', 'PlantDoc-Dataset', 'Images_LEGO_Bricks', 'Art_Types_Classification', 'Weather_Type_Dataset', 'Simpsons_Characters_Data', 'Intel_Image_Classification', 'places365_small', 'House_Room_Images', 'UIUC_Sports_Event_Dataset', 'Land-Use_Scene_Classification', 'ASL_Alphabets_Dataset', 'Yoga-82', 'Russian_Letter_Dataset', 'UMNIST', 'iFood2019', 'ePillID_data', 'Oxford_Buildings', 'Texture_Dataset', 'electronic-components', 'Hurricane_Damage_Dataset', 'chest_xray', 'PAD-UFES-20', 'Brain_Tumor_Dataset', 'Kannada-MNIST', 'Breast_Ultrasound', 'BookCover30', 'boat-types-recognition', 'rock-classification', 'dermnet', 'dragon-ball-super-saiyan-dataset', 'concrete-crack', 'Malacca_Historical_Buildings', 'African_countries', 'skin-cancer-mnist-ham10000', 'FaceMask_Dataset', 'watermarked-not-watermarked-images', 'Fish_Dataset', 'deepweedsx', 'ip02-dataset', 'planets-and-moons-dataset-ai-in-space', 'polish-craft-beer-labels', 'the-kvasircapsule-dataset', 'surface_defect_database', 'minerals-identification-classification', 'colorectal-histology-mnist', '100Sports', 'SurgicalTools', 'MechanicalTools', 'Galaxy10', 'Stanford_Online_Products', 'NWPU-RESISC45']
# with open("/lab/tmpig20c/u/zmurdock/shell-datasets/shell_list.txt") as f:
#     for line in f.readlines():
#         original_class_list.append(line.strip().split()[1])
for class_name in original_class_list:
    path = class_name + ".pth"
    val_path = class_name + "_val.pth"
    features_path['train'].append(path)
    features_path['val'].append(val_path)

index = np.linspace(0, 100, 101)
# np.random.shuffle(index)


train_classes = []
train_sizes = []
gmmc_training_time = 0
for t in range(len(features_path['train'])):
    if t not in index[:task_num]:
        continue
    train_path = '/lab/tmpig8c/u/dw-code/ShELL_GPU/Pytorch/birds/GMMC_src/100_dataset_sample_features/' + features_path['train'][t]
    task_name = features_path['train'][t][:-4]
    log(f"t{task_num}_clu{cluster_num}", f"start training task{t}: {task_name}", write_time=True)
    train_features = list(torch.load(train_path))
    
    # get n_class for each task 
    train_labels = []
    for i, label in train_features:
        train_labels.append(label)
    task_n_classes = len(set(train_labels))
    
    sample_class_features = [[] for _ in range(task_n_classes)]
    
    
    # append features and compute sample mean for each class for current task 
    for feat, label in train_features:
        sample_class_features[label].append(feat)
    log(f"t{task_num}_clu{cluster_num}", f"task{t}({task_name}) has {task_n_classes} classes and train_size={len(train_features)}")
    train_classes.append(task_n_classes)
    train_sizes.append(len(train_features))
    X = []
    for label, class_features in enumerate(sample_class_features):
        X.extend(class_features)
    X = np.array(X)
    start_time = time.time()
    high.start_counters([events.PAPI_SP_OPS,])
    tmapper.fit_task(task_num, cluster_num, t, X)
    x = high.stop_counters()
    # print(x)
    gmmc_training_time += (time.time() - start_time) 
    log(f"t{task_num}_clu{cluster_num}", f"finish training task{t}: {task_name}", write_time=True)
# log(f"t{task_num}_clu{cluster_num}", f"training time: {gmmc_training_time}")
# test_classes = []
# test_sizes = []
# accs = []
# fp_ops = []
# sp_ops = []
# dp_ops = []
# eval_time = []
# accumulate_time = 0
# for t in range(len(features_path['val'])):
#     if t not in index[:task_num]:
#         continue
#     # load features and get task name 
#     val_path = '/lab/tmpig8c/u/dw-code/ShELL_GPU/Pytorch/birds/GMMC_src/100_dataset_sample_features/' + features_path['val'][t]
#     task_name = features_path['val'][t][:-4]
#     log(f"t{task_num}_clu{cluster_num}", f"start evaluate task{t}: {task_name}", write_time=True)
#     val_features = list(torch.load(val_path))
#     val_feats = []
#     val_labels = set({})
#     for feature, label in val_features:
#         val_labels.add(label)
#         val_feats.append(feature)
#     val_feats = np.array(val_feats)
#     num_samples = val_feats.shape[0]          
#     targets_task = [t] * num_samples
#     for val_feat in val_feats:
#         # start_time = time.time()
#         # high.start_counters([events.PAPI_FP_OPS,])
#         # tmapper.predict_task(val_feat)
#         # x = high.stop_counters()
#         # eval_time.append(time.time() - start_time + accumulate_time)
#         # accumulate_time = eval_time[-1]
#         # fp_ops += x
#         high.start_counters([events.PAPI_SP_OPS,])
#         tmapper.predict_task(val_feat)
#         x = high.stop_counters()
#         sp_ops += x
#         high.start_counters([events.PAPI_DP_OPS,])
#         tmapper.predict_task(val_feat)
#         x = high.stop_counters()
#         dp_ops += x
#     break
# log(f"t{task_num}_clu{cluster_num}", f"class index {index[:task_num]}")
# log(f"t{task_num}_clu{cluster_num}", f"eval time: {eval_time}")
# log(f"t{task_num}_clu{cluster_num}", f"fp: {fp_ops}")
# log(f"t{task_num}_clu{cluster_num}", f"sp: {sp_ops}")
# log(f"t{task_num}_clu{cluster_num}", f"dp: {dp_ops}")


#     # preds_task = [int(tmapper.predict_task(val_feats[i,...])) for i in range(num_samples)]
#     # acc_task = sum([targets_task[i]==preds_task[i] for i in range(len(preds_task))])/len(preds_task)
#     # test_classes.append(len(val_labels))
#     # test_sizes.append(len(val_features))
#     # log(f"t{task_num}_clu{cluster_num}", f"task{t} {task_name} acc is {acc_task}", write_time=True)
#     # accs.append(acc_task)
# # index[:task_num].sort()
# # log(f"t{task_num}_clu{cluster_num}", f"class index {index[:task_num]}")
# # log(f"t{task_num}_clu{cluster_num}", f"train_class numbers {train_classes}")
# # log(f"t{task_num}_clu{cluster_num}", f"train_sizes numbers {train_sizes}")
# # log(f"t{task_num}_clu{cluster_num}", f"test_class numbers {test_classes}")
# # log(f"t{task_num}_clu{cluster_num}", f"test_sizes numbers {test_sizes}")
# # log(f"t{task_num}_clu{cluster_num}", accs, write_time=True)
# # correct = 0
# # total = 0
# # for i in range(len(accs)):
# #     correct += accs[i] * test_sizes[i]
# #     total += test_sizes[i]
# # log(f"t{task_num}_clu{cluster_num}", correct/total, write_time=True)