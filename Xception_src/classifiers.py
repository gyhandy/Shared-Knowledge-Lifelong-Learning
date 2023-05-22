import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math

import sys
import torchvision
from torchvision import datasets, models, transforms

from Xception_src.Conv_BP_layer_prototype import *

class Xception_Baseline_multiple_task(nn.Module):
    def __init__(self,num_classes, num_tasks):
        super(Xception_Baseline_multiple_task,self).__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        import timm
        model = timm.create_model('xception',pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False
        self.base = nn.Sequential(*list(model.children())[:-1])
        self.feat_size = 2048
        for i in range(0,self.num_tasks):
            setattr(self, "fc%d" % i, nn.Linear(self.feat_size, self.num_classes[i]))

    def forward(self, x, task_num):
        x = self.base(x)
        x = torch.flatten(x, 1)

        clf_outputs = getattr(self, "fc%d" % task_num)(x)
            
        return clf_outputs
    
class Xception_Baseline_single_task(nn.Module):
    def __init__(self, num_class):
        super(Xception_Baseline_single_task,self).__init__()
        self.num_class = num_class
        import timm
        model = timm.create_model('xception',pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False
        self.base = nn.Sequential(*list(model.children())[:-1])
        self.feat_size = 2048
        self.fc = nn.Linear(self.feat_size, self.num_class)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)

        clf_outputs = self.fc(x)
            
        return clf_outputs
    
class Xception_extend_single_task(nn.Module):
    def __init__(self, num_class):
        super(Xception_extend_single_task,self).__init__()
        self.num_class = num_class
        import timm
        model = timm.create_model('xception',pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False
        self.base = nn.Sequential(*list(model.children())[:-1])
        self.feat_size = 2048
        self.fc1 = nn.Linear(self.feat_size, 2048)
        self.fc2 = nn.Linear(2048, self.num_class)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        clf_outputs = self.fc2(x)
            
        return clf_outputs

def add_ConvBP_prototype(model):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            channel_num = layer.out_channels
            model._modules[name] = nn.Sequential(layer, ConvBP_layer_prototype(channel_num, epsilon=0.2))
        add_ConvBP_prototype(layer)
    return model

def add_Convbias(model):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            channel_num = layer.out_channels
            model._modules[name] = nn.Sequential(layer, ConvBias_layer(channel_num))
        add_Convbias(layer)
    return model

class Xception_TB(nn.Module):
    def __init__(self, num_class):
        super(Xception_TB,self).__init__()
        self.num_class = num_class
        import timm
        model = timm.create_model('xception',pretrained=True)
        for param in model.parameters():
            param.require_grad = False

        self.base = nn.Sequential( *(list(add_Convbias(model).children())[:-1]) )
        self.feat_size = 2048
        self.fc = nn.Linear(self.feat_size, self.num_class)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)

        clf_outputs = self.fc(x)
            
        return clf_outputs
    
class Xception_BP(nn.Module):
    def __init__(self, num_class):
        super(Xception_BP,self).__init__()
        self.num_class = num_class
        import timm
        model = timm.create_model('xception',pretrained=True)

        self.base = nn.Sequential( *list(add_ConvBP_prototype(model).children())[:-1] )
        self.feat_size = 2048
        self.fc = nn.Linear(self.feat_size, self.num_class)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)

        clf_outputs = self.fc(x)
            
        return clf_outputs
    
class Resnet_TB(nn.Module):
    def __init__(self, num_class):
        super(Resnet_TB,self).__init__()
        self.num_class = num_class
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.require_grad = False

        self.base = nn.Sequential( *(list(add_Convbias(model).children())[:-1]) )
        self.feat_size = 512
        self.fc = nn.Linear(self.feat_size, self.num_class)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)

        clf_outputs = self.fc(x)

        return clf_outputs
    