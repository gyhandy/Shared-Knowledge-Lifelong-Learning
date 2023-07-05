import os
import argparse
from copy import deepcopy

import torch
import torch.optim as optim
import timm

from Xception_src.classifiers import *
from Xception_src.Conv_BP_layer_prototype import *
from dataset.dataloader_reader import load_dataloader
import gmmc_grid_search.TaskMappers.proto_mapper as pmap
from maha_src.mahalanobis import * 
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default='./result/')
    parser.add_argument('--weight', type=str, default='./weight/')
    parser.add_argument('--data', type=str, default="/lab/tmpig15b/u/")
    parser.add_argument('--method', type=str, default="BB_SKILL", help="Linear_SKILL for linear feature dataset with \
                        fc layer. BB_SKILL for imagedataset with BB implementation. If you want to test the fc performance with a \
                        image folder, simply type Linear")
    parser.add_argument('--task_mapper', type=str, default="GMMC")
    parser.add_argument('--n_c', type=int, default=25)
    parser.add_argument('--mag', type=int, default=5)
    parser.add_argument('--prediction', type=str, default='./prediction/')
    parser.add_argument('--activation_size', type=int, default=2048)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.result):
        os.mkdir(args.result)
    if not os.path.exists(args.weight):
        os.mkdir(args.weight)
    if not os.path.exists(args.prediction):
        os.mkdir(args.prediction)
    # @TODO You may want to use your own code for dataset, here we provide datasets we presented in the paper
    ##############################################################################################################
    if args.method == "Linear_SKILL":
        train_datasets, test_datasets, train_loaders, test_loaders = load_dataloader(-1, input_type="features")
    else:
        train_datasets, test_datasets, train_loaders, test_loaders = load_dataloader(-1, input_type="original")
    num_task = len(train_datasets)
    #############################################################################################################
    for i in range(num_task):
        train_dataset = train_datasets[i]
        test_dataset = test_datasets[i]
        train_loader = train_loaders[i]
        test_loader = test_loaders[i]

        task_name = train_dataset.dataset_name # @TODO name for your ith dataset, you can set it with your own method
        train_num_class = train_dataset.num_classes # @TODO number of classes of your ith dataset, you can set it with your own method

        if args.method == "Linear":
            model = timm.create_model('xception',pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = torch.nn.Linear(in_features=args.activation_size, out_features=train_num_class, bias=True)
        elif args.method == "Linear_SKILL":
            model = torch.nn.Linear(in_features=args.activation_size, out_features=train_num_class, bias=True)
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
            train_acc, train_loss = train(train_loader, model, optimizer, device)
            model.eval()
            val_acc = eval(model, test_loader, device)
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

    # Now we can do GMMC, you can also replaced with a perfect task oracle
    if args.task_mapper == "GMMC":
        # @TODO you need to create a feature dataset which is the activation from your backbone network
        feature_train_datasets, feature_val_datasets, feature_train_loaders, feature_val_loaders = load_dataloader(-1, shuffle_test=False, input_type="features")

        total_classes = [train_dataset.num_classes for train_dataset in feature_train_datasets]
        class_list = [train_dataset.dataset_name for train_dataset in feature_train_datasets]

        for i in range(len(class_list)):
            if args.method == "Linear_SKILL":
                fc = torch.nn.Linear(in_features=args.activation_size, out_features=total_classes[i], bias=True)
                state_dict = torch.load(os.path.join(args.weight, f"{args.method}_{class_list[i]}.pth"), map_location=device)
                with torch.no_grad():
                    fc.weight.copy_(state_dict['weight'])
                    fc.bias.copy_(state_dict["bias"])
                fc.to(device)
                fc.eval()

                inference_result = inference(fc, feature_val_loaders[i])
                torch.save(inference_result, os.path.join(args.prediction, f"{args.method}_{class_list[i]}.pth"))

            elif args.method == "BB_SKILL":
                BB_model = Xception_TB(total_classes[i])
                original_state_dict = BB_model.state_dict()

                state_dict = torch.load(os.path.join(args.weight, f"{args.method}_{class_list[i]}.pth"), map_location=device)
                for param in original_state_dict:
                    if param[-6:] == "1.bias" or param == "fc.weight" or param == "fc.bias":
                        original_state_dict[param] = state_dict[param]
                BB_model.load_state_dict(state_dict)
                BB_model.to(device)
                BB_model.eval()

                BB_result = inference(BB_model, test_loaders[i])
                torch.save(BB_result, os.path.join(args.prediction, f"{args.method}_{class_list[i]}.pth"))

        prediction_results =[]
        for i in range(len(class_list)):
            prediction_result = torch.load(os.path.join(args.prediction, f"{args.method}_{class_list[i]}.pth"))
            prediction_results.append(prediction_result)

        # setting for gmmc
        Final_accs = []
        with open("./gmmc_grid_search/Shell_8dset.yaml", 'r') as stream:
            try:
                gmmc_args=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                log(args.result, str(i), str(exc), write_time=True)
        gmmc_args = setup(gmmc_args)
        seed_torch(gmmc_args['seed'])
        torch.set_num_threads(gmmc_args['num_threads'])
        pin_memory=False
        tmapper = pmap.ProtoTaskMapper(gmmc_args['task_mapper_params'])

        def write_result(filename, result):
            with open(filename, "a") as f:
                f.write(str(result)[1:-1])
                f.write("\n")

        def test_task_mapping(currentTask, task_mapper):
            final_accs = [0 for _ in range(num_task)]
            for task in range(currentTask+1):
                t = task
                num_samples = len(feature_val_datasets[t])
                targets_task = [t] * num_samples
                preds_task = []
                for val_feat, _ in feature_val_datasets[t]:
                    preds_task.append(int(task_mapper.predict_task(val_feat)))
                GMMC_result = [targets_task[i]==preds_task[i] for i in range(len(preds_task))]
                final_result = [GMMC_result[i] and prediction_results[task][i] for i in range(len(GMMC_result))]
                final_accs[t] = (sum(final_result)/len(GMMC_result))

            return final_accs

        final_accs = [[] for _ in range(num_task)]
        for t in range(num_task):
            X = []
            for i in range(len(feature_train_datasets[t])):
                X.append(feature_train_datasets[t][i][0].unsqueeze(0))
            X = torch.cat(X, dim=0).numpy()
            tmapper.fit_task(args.n_c, t, X)
            final_acc = test_task_mapping(t, tmapper)
            write_result(os.path.join(args.result, f"gmmc_{args.n_c}.csv"), final_acc)
            if t == num_task-1:
                log(args.result, "final", f"final acc is {np.average(final_acc)}", args.task_mapper, write_time=True)


    elif args.task_mapper == "MAHA":

        # @TODO you need to create a feature dataset which is the activation from your backbone network
        feature_train_datasets, feature_val_datasets, feature_train_loaders, feature_val_loaders = load_dataloader(-1, shuffle_test=False, input_type="features")

        total_classes = [train_dataset.num_classes for train_dataset in feature_train_datasets]
        class_list = [train_dataset.dataset_name for train_dataset in feature_train_datasets]

        for i in range(len(class_list)):
            if args.method == "Linear_SKILL":
                fc = torch.nn.Linear(in_features=args.activation_size, out_features=total_classes[i], bias=True)
                state_dict = torch.load(os.path.join(args.weight, f"{args.method}_{class_list[i]}.pth"), map_location=device)
                with torch.no_grad():
                    fc.weight.copy_(state_dict['weight'])
                    fc.bias.copy_(state_dict["bias"])
                fc.to(device)
                fc.eval()

                inference_result = inference(fc, feature_val_loaders[i])
                torch.save(inference_result, os.path.join(args.prediction, f"{args.method}_{class_list[i]}.pth"))

            elif args.method == "BB_SKILL":
                BB_model = Xception_TB(total_classes[i])
                original_state_dict = BB_model.state_dict()

                state_dict = torch.load(os.path.join(args.weight, f"{args.method}_{class_list[i]}.pth"), map_location=device)
                for param in original_state_dict:
                    if param[-6:] == "1.bias" or param == "fc.weight" or param == "fc.bias":
                        original_state_dict[param] = state_dict[param]
                BB_model.load_state_dict(state_dict)
                BB_model.to(device)
                BB_model.eval()

                BB_result = inference(BB_model, test_loaders[i])
                torch.save(BB_result, os.path.join(args.prediction, f"{args.method}_{class_list[i]}.pth"))

        prediction_results =[]
        for i in range(len(class_list)):
            prediction_result = torch.load(os.path.join(args.prediction, f"{args.method}_{class_list[i]}.pth"))
            prediction_results.append(prediction_result)

        def write_result(filename, result):
            with open(filename, "a") as f:
                f.write(str(result)[1:-1])
                f.write("\n")

        final_accs = [[] for _ in range(num_task)]
        for t in range(num_task):
            precision, sample_class_mean, num_classes = compute_mabalanobis_stats(feature_train_datasets[:t+1], args.mag, device)
            final_acc = run_mahalanobis(feature_val_datasets, precision, sample_class_mean, prediction_results, device)
            final_accs[t] = final_acc
            write_result(os.path.join(args.result, f"maha_{args.mag}.csv"), final_acc)
            if t == num_task-1:
                log(args.result, "final", f"final acc is {np.average(final_acc)}", args.task_mapper, write_time=True)
