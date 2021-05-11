import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)  # input32 output 28
        self.pool1 = nn.MaxPool2d(2,2) # input 28 output 14
        self.conv2 = nn.Conv2d(6,16,5) # input 14 output 10
        self.pool2 = nn.MaxPool2d(2,2) # 5 * 5 * 16
        self.fc1   = nn.Linear(5*5*16,120)
        self.fc2   = nn.Linear(120,84)
        self.fc3   = nn.Linear(84,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x  = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

