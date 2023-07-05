# Shared-Knowledge-Lifelong-Learning
[TMLR] Lightweight Learner for Shared Knowledge Lifelong Learning
### [Project Page(pending)](http://ilab.usc.edu/) | [Video(pending)](https://youtu.be/) | [Paper](https://arxiv.org/pdf/2305.15591.pdf)

> **Lightweight Learner for Shared Knowledge Lifelong Learning** <br>
> Yunhao Ge, Yuecheng Li, Di Wu, Ao Xu, Adam M. Jones, Amanda Sofie Rios, Iordanis Fostiropoulos, shixian wen, Po-Hsuan Huang, Zachary William Murdock, Gozde Sahin, Shuo Ni, Kiran Lekkala, Sumedh Anand Sontakke, Laurent Itti <br>
> *Transactions on Machine Learning Research*

<div align="center">
    <img src="./docs/Fig-3.png" alt="Editor" width="500">
</div>

**Figure:** *Algorithm design. Top: overall pipeline, where agents are deployed in different regions to learn their own tasks. Subsequently, learned knowledge is shared among all agents. Bottom: Zoom into the details of each agent, with 4 main roles: 1) Training: agents use a common pre-trained and frozen backbone, stored in ROM memory at manufacturing time (gray trapezoid with lock symbol). The backbone allows the agent to extract compact representations from inputs (e.g., with an xception backbone, the representation is a latent vector of 2048 dimensions, and inputs are 299 × 299 RGB images). Each agent learns a task-specific head (red triangle) for each new task. A head consists of the last fully-connected layer of the network plus our proposed LL beneficial biasing units (BB) that provide task-dependent tuning biases to all neurons in the network (one float number per neuron). During training, each agent also learns a GMMC or Mahalanobis task anchor which will form a task mapper. 2) Share knowledge with other agents: each agent shares the learned task-specific head, Beneficial Bias (BB), and GMMC module (or training images for Mahalanobis) with all other agents. 3) Receive knowledge from other agents: each agent receives different heads and GMMC/Mahalanobis task mapper anchors from other agents. All heads are stored in a head bank and all task anchors are consolidated to form a task mapper. 4) Testing: At test time, an input is first processed through the task mapper. This outputs a task ID, used to load up the corresponding head (last layer + beneficial biases) from the bank. The network is then equipped with the correct head and is run on the input to produce an output.*


We have proposed a new framework for shared-knowledge, parallelized LL. On a new, very challenging SKILL-102 dataset, we find that this approach works much better than previously SOTA baselines, and is much faster. Scaling to > 500 difficult tasks like the ones in our new SKILL-102 dataset seems achievable with the current implementation.

<div align="center">
    <img src="./docs/Fig-skill-1.png" alt="Editor" width="500">
</div>

**Figure:** *SKILL vs. related learning paradigms. a) Multi-task learning (Caruana, 1997): one agent learns all tasks at the same time in the same physical location. b) Sequential Lifelong Learning (S-LL) (Li & Hoiem, 2017): one agent learns all tasks sequentially in one location, deploying LL-specific machinery to avoid task interference. c) Federated learning (McMahan et al., 2017): multiple agents learn the same task in different physical locations, then sharing learned knowledge (parameters) with a center agent. d) Our SKILL: different S-LL agents in different physical regions each learn tasks, and learned knowledge is shared among all agents, such that finally all agents can solve all tasks. Bottom-right table: Strengths & weaknesses of each approach*

## Getting starged

### Installation(single file)
```
git clone https://github.com/gyhandy/Shared-Knowledge-Lifelong-Learning.git
wget https://ilab.usc.edu/andy/skill-dataset/skill/SKILL-Dataset-backend.zip
unzip SKILL-Dataset-backend.zip
```
### download file(multiple files) (either way will work)
```
git clone https://github.com/gyhandy/Shared-Knowledge-Lifelong-Learning.git
wget https://ilab.usc.edu/andy/skill-dataset/seperate-v/skill-dataset.z01
wget https://ilab.usc.edu/andy/skill-dataset/seperate-v/skill-dataset.z02
wget https://ilab.usc.edu/andy/skill-dataset/seperate-v/skill-dataset.zip
unzip skill-dataset.zip
```

### General directory structure

- `dataset/` contains code for declare `train_datasets`, `val_datasets`, `train_loaders`, and `val_loaders` for the DCT dataset, each is a list of 107 datasets contains in the DCT. You can also define your own `train_datasets`, `val_datasets`, `train_loaders`, and `val_loaders` for your own datasets. Place to change is commented in the main code

- `Xception_src/` contains models and specific customized layer used in the experiment
 
- `gmmc_grid_search/` contains the GMMC classifier

- `main.py/` The main code needs to be run

### Usage

#### argument

- `--result` the result path will store all the logs

- `--weight` the weight path which store the weight of the classifiers

- `--prediction` the folder to store the prediction results for each instance in the dataset

- `--data` the path to store the data, **this should be where you unzip before**

- `--method` `BB_SKILL` refers to the BB network mentioned in the paper and `Linear_SKILL` refers to the linear classifier with a fixed backbone

- `--task_mapper` types of task mapper, either `GMMC` or `MAHA`

- `--n_c` number of clusters used for `GMMC`

- `--activation_size` the size of the activation vector (e.g. resnet18 has size 512 and resnet 50 has size 2048)

#### Sample run

- To run a BB network with GMMC
```
python main.py
```

- To run a Linear Classifier with GMMC
```
python main.py --method Linear_SKILL
```
