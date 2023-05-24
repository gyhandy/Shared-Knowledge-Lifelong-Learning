import torch
import random
import numpy as np 
import time
import torch.nn as nn
import copy
from sklearn.covariance import EmpiricalCovariance

def compute_mabalanobis_stats(feature_train_datasets, mag, device):
    
    sample_class_mean = {train_dataset.dataset_name: [] for train_dataset in feature_train_datasets}
    all_feats = []
    num_classes = []
    
    for t in range(len(feature_train_datasets)):
        # load features and get task name 
        task_name = feature_train_datasets[t].dataset_name
        #print(f'{task_name}: start')

        train_features = []
        for i in range(len(feature_train_datasets[t])):
            train_features.append(feature_train_datasets[t][i][0].unsqueeze(0))
        train_features = list(torch.cat(train_features, dim=0).numpy())
        # get n_class for each task 
        train_labels = []
        for i in range(len(feature_train_datasets[t])):
            train_labels.append(feature_train_datasets[t][i][0].unsqueeze(0))
        train_labels = list(torch.cat(train_labels, dim=0).numpy())

        task_n_classes = len(set(train_labels))
        num_classes.append(task_n_classes)

        #print(task_n_classes)
        sample_class_features = [[] for _ in range(task_n_classes)]
        sample_class_mean[task_name] = [0 for _ in range(task_n_classes)]
        # append features and compute sample mean for each class for current task 
        for feat, label in zip(train_features, train_labels):
            sample_class_features[label].append(feat)
        for label in range(task_n_classes):
            sample_class_mean[task_name][label] = np.mean(sample_class_features[label], axis=0)
        X = []
        X_norm = []
        for label, class_features in enumerate(sample_class_features):
            if len(class_features) > mag:
                sampled_features = random.sample(class_features, mag)
                X_norm.extend(sampled_features - sample_class_mean[task_name][label])
                X.extend(sampled_features)
            else:                
                X_norm.extend(class_features - sample_class_mean[task_name][label])
                X.extend(class_features)
        X = np.array(X)
        X_norm = np.array(X_norm)
        all_feats.extend(X_norm)
    
    print(np.array(all_feats).shape)

    group_lasso = EmpiricalCovariance(assume_centered=False)
    group_lasso.fit(np.array(all_feats))
    precision = group_lasso.precision_

    precision = torch.tensor(precision, dtype=torch.float32).to(device)     
    return precision, sample_class_mean, num_classes

def check_acc(a, b):
    correct = 0
    for i in range(len(a)):
        if a[i] == True and b[i] == True:
            correct += 1 
    return correct

def run_mahalanobis(feature_val_datasets, precision, sample_class_mean, prediction_results, device):
    final_accs = [0 for _ in range(len(feature_val_datasets))]
    bs = 256 
    for t in range(len(feature_val_datasets)):

        # load features 
        val_features = []
        for i in range(len(feature_val_datasets[t])):
            val_features.append(feature_val_datasets[t][i][0].unsqueeze(0))
        act = list(torch.cat(val_features, dim=0).numpy())
        num_samples = act.shape[0]

        #print(f'{task_name}: start with {num_samples}')
        preds_task = []
        for b in range(0, num_samples, bs):
            if b+bs > num_samples:
                batch_feature = act[b:]
            else:
                batch_feature = act[b:b+bs]
            batch_feature = torch.tensor(batch_feature, dtype=torch.float32).to(device)
            task_gaussian_score = 0
            for i, (task, task_means) in enumerate(sample_class_mean.items()):
                gaussian_score = 0 
                for j, sample_mean in enumerate(sample_class_mean[task]):
                    zero_f = batch_feature - torch.tensor(sample_mean, dtype=torch.float32).to(device) 
                    term_gau = -0.5*torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                    if j == 0:
                        gaussian_score = term_gau.view(-1,1)
                    else:
                        gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
                            
                gaussian_score = gaussian_score.max(1)[0]
                    
                if i == 0:
                    task_gaussian_score = gaussian_score.view(-1,1)
                else:
                    task_gaussian_score = torch.cat((task_gaussian_score, gaussian_score.view(-1,1)), 1)       
            batch_prediction = task_gaussian_score.max(1)[1].cpu().numpy()
            preds_task.extend(batch_prediction)
                        
        targets_task = [t] * num_samples
        maha_pred = [targets_task[i]==preds_task[i] for i in range(num_samples)]
        final_accs[t] = (check_acc(prediction_results[t], maha_pred)/num_samples)
    return final_accs