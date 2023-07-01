# Shared-Knowledge-Lifelong-Learning
[TMLR] Lightweight Learner for Shared Knowledge Lifelong Learning

# download file

```
wget https://ilab.usc.edu/andy/skill-dataset/skill/SKILL-Dataset-backend.zip
unzip SKILL-Dataset-backend.zip
```

## General directory structure

- `dataset/` contains code for declare `train_datasets`, `val_datasets`, `train_loaders`, and `val_loaders` for the DCT dataset, each is a list of 107 datasets contains in the DCT. You can also define your own `train_datasets`, `val_datasets`, `train_loaders`, and `val_loaders` for your own datasets. Place to change is commented in the main code

- `Xception_src/` contains models and specific customized layer used in the experiment
 
- `gmmc_grid_search/` contains the GMMC classifier

- `main.py/` The main code needs to be run

## Usage

### argument

- `--result` the result path will store all the logs

- `--weight` the weight path which store the weight of the classifiers

- `--prediction` the folder to store the prediction results for each instance in the dataset

- `--data` the path to store the data

- `--method` `BB_SKILL` refers to the BB network mentioned in the paper and `Linear_SKILL` refers to the linear classifier with a fixed backbone

- `--task_mapper` types of task mapper, either `GMMC` or `MAHA`

- `--n_c` number of clusters used for `GMMC`

- `--activation_size` the size of the activation vector (e.g. resnet18 has size 512 and resnet 50 has size 2048)

### Sample run

- To run a BB network with GMMC
```
python main.py
```

- To run a Linear Classifier with GMMC
```
python main.py --method Linear_SKILL
```
