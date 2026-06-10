
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F



def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, nb_inputs, nb_outputs, hidden, init_w=5e-1):
        super(Actor, self).__init__()
        self.fc = nn.ModuleList()
        for layer in range(len(hidden)+1):
            if layer == 0:
                self.fc.append(nn.Linear(nb_inputs, hidden[0]))
            elif layer == len(hidden):
                self.fc.append(nn.Linear(hidden[layer-1], nb_outputs))
            else:
                self.fc.append(nn.Linear(hidden[layer-1], hidden[layer]))
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in range(len(self.fc)):
            if layer == len(self.fc)-1:
                self.fc[layer].weight.data.uniform_(-init_w, init_w)
            else:
                self.fc[layer].weight.data = fanin_init(self.fc[layer].weight.data.size())

    def forward(self, x):
        out = x
        for layer in range(len(self.fc)):
            out = self.fc[layer](out)
            if layer < len(self.fc)-1:
                out = self.relu(out)
        return out

    def print_num_params(self): 
        total_params = sum(p.numel() for p in self.parameters())
        total_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f'Total number of actor parameters: {total_params}')
        print(f'Total number of actor trainable parameters: {total_params_trainable}')


class Critic(nn.Module):
    def __init__(self, nb_inputs, nb_actions, hidden, init_w=5e-1):
        super(Critic, self).__init__()
        self.fc=nn.ModuleList()
        for layer in range(len(hidden)+1):
            if layer == 0:
                self.fc.append(nn.Linear(nb_inputs, hidden[0]))
            elif layer == math.floor(len(hidden)/2):
                self.layer_num_for_action = layer
                self.fc.append(nn.Linear(hidden[layer-1]+nb_actions, hidden[layer]))
            elif layer == len(hidden):
                self.fc.append(nn.Linear(hidden[layer-1], 1))
            else:
                self.fc.append(nn.Linear(hidden[layer-1], hidden[layer]))
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in range(len(self.fc)):
            if layer == len(self.fc)-1:
                self.fc[layer].weight.data.uniform_(-init_w, init_w)
            else:
                self.fc[layer].weight.data = fanin_init(self.fc[layer].weight.data.size())

    def forward(self, xs):
        x, price, a = xs
        out = torch.cat([x, price], -1)
        for layer in range(len(self.fc)):
            if layer == self.layer_num_for_action:
                out = self.fc[layer](torch.cat([out, a], -1))
            else:
                out = self.fc[layer](out)
            if layer < len(self.fc)-1:
                out = self.relu(out)
        return out

    def print_num_params(self): 
        total_params = sum(p.numel() for p in self.parameters())
        total_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of critic parameters: {total_params}')
        print(f'Total number of critic trainable parameters: {total_params_trainable}')
