# Shared-Knowledge-Lifelong-Learning
[TMLR] Lightweight Learner for Shared Knowledge Lifelong Learning

## Train BB network
To train an Xception with Benificial Bias and custom head, simply run
```
cd Xception_src
python main.py 0:0:BPN:<batch_size>
```
You may want to change line 114 to 116 to get the correct dataset and loader
You can also change line 30 and 31 to change logging path

## Train FC head
To simply train a FC head
```
cd fc_train
python initial_training.py
```

## Train GMMC
```
cd gmmc_grid_search
python gmmc_xception.py
```

just move gmmc_grid_search/gmmc_xception.py to main.py