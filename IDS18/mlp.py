import argparse
import os
import pickle
import sys
import warnings
from datetime import datetime
import torch
import torch.nn as nn



from torch.nn import MSELoss
from torch.optim import Adam
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
"""
This code was refered from the https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
"""

class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 100)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(100, 1024)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(1024, 500)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.hidden4 = Linear(500, 250)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        self.hidden5 = Linear(250, 50)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        self.hidden6 = Linear(50, 1)
        kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        self.act6 = Sigmoid()
#         xavier_uniform_(self.hidden3.weight)
#         #self.double()
#         self.act7 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = X.float()
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        X = self.act4(X)
        X = self.hidden5(X)
        X = self.act5(X)
        X = self.hidden6(X)
        X = self.act6(X)
        return X
