import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.function import once_differentiable
import math
import numpy as np 

"""
Is It correct to just add all elements for output_grad when doing backward
"""

class ConvBP_prototype(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod

    def forward(ctx, input, weight_memory_blocks_task, memory_blocks_task, epsilon=0.2):

        ctx.epsilon = epsilon
        
        memory_blocks_task_middle = memory_blocks_task.view(-1,
                                                              memory_blocks_task.size(
                                                                  1) * memory_blocks_task.size(2))

        weight_memory_blocks_task_middle = weight_memory_blocks_task.view(weight_memory_blocks_task.size(0) * weight_memory_blocks_task.size(1), -1)
      

        ctx.save_for_backward(input, weight_memory_blocks_task, memory_blocks_task)
        input = input.view(input.shape[0], input.shape[2], input.shape[3], input.shape[1])
        output = input + memory_blocks_task_middle.mm(weight_memory_blocks_task_middle)
        output = output.view(output.shape[0], output.shape[3], output.shape[1], output.shape[2])
        return output


    @staticmethod
    def backward(ctx, grad_output):

        input, weight_memory_blocks_task , memory_blocks_task = ctx.saved_variables

        epsilon = ctx.epsilon

        grad_input = grad_weight_memory_blocks_task = grad_memory_blocks_task = None

        grad_input = grad_output
        sum_output = grad_output.sum((2, 3))

        grad_memory_blocks_task = sum_output.mm(weight_memory_blocks_task.view(weight_memory_blocks_task.size(0)*weight_memory_blocks_task.size(1),weight_memory_blocks_task.size(2)).t())
        grad_memory_blocks_task = (grad_memory_blocks_task.sum(0)).view(memory_blocks_task.size(0),memory_blocks_task.size(1),memory_blocks_task.size(2))
        grad_memory_blocks_task = torch.sign(grad_memory_blocks_task)*epsilon
        grad_weight_memory_blocks_task = (memory_blocks_task.view(memory_blocks_task.size(0),memory_blocks_task.size(1)*memory_blocks_task.size(2)).t()).mm(sum_output.sum(0).unsqueeze(0))
        grad_weight_memory_blocks_task = grad_weight_memory_blocks_task.view(weight_memory_blocks_task.size(0),weight_memory_blocks_task.size(1),weight_memory_blocks_task.size(2))
        
        return grad_input, grad_weight_memory_blocks_task, grad_memory_blocks_task, None, None, None


class ConvBP_layer_prototype(nn.Module):
    def __init__(self, feature_num, epsilon=0.2, memory_block=50*50):
        super(ConvBP_layer_prototype, self).__init__()

        self.memory_block = memory_block
        self.epsilon = epsilon
        self.input_features = feature_num
        self.output_features = feature_num
        input_features_memory_units = 1
        
        self.weight_memory_blocks_task = nn.Parameter(torch.Tensor(input_features_memory_units, memory_block, feature_num))
        self.memory_blocks_task = nn.Parameter(torch.Tensor(1, input_features_memory_units, memory_block))
        self.reset_parameters()

    def reset_parameters(self):
        stdv_weight_memory_blocks_task = 1./math.sqrt(self.weight_memory_blocks_task.size(0)*self.weight_memory_blocks_task.size(1))
        self.weight_memory_blocks_task.data.uniform_(-stdv_weight_memory_blocks_task,stdv_weight_memory_blocks_task)
        self.memory_blocks_task.data.zero_()

    def forward(self, input):
        return ConvBP_prototype.apply(input,self.weight_memory_blocks_task, self.memory_blocks_task, self.epsilon)


class ConvBias_layer(nn.Module):
    def __init__(self, feature_num):
        super(ConvBias_layer, self).__init__()
        self.input_features = feature_num
        self.bias = nn.Parameter(torch.Tensor(feature_num))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.bias.size(0))
        self.bias.data.zero_()

    def forward(self, input):
        # print(input.shape[0]* input.shape[2]* input.shape[3]* input.shape[1])
        input = input.view(input.shape[0], input.shape[2], input.shape[3], input.shape[1])
        output = input + self.bias
        output = output.view(output.shape[0], output.shape[3], output.shape[1], output.shape[2])
        return output