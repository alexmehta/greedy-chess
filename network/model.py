import torch.nn as nn
from network_to_san import Regressor
class Model(nn.Module):
    """Base model"""
    def __init__(self):
        super(Model, self).__init__()
        self.input  = nn.Sequential(nn.Conv2d(8,kernel_size=(3,3)),nn.Conv2d(16,kernel_size=(3,3)),nn.Flatten())
        self.output = Regressor()
    def forward(self, x):
        x = self.input(x)
        print(x.shape)
        return self.output(x)