import torch
from torch import nn
from torch.nn import functional as F

# maybe try a CNN instead
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        self.model = nn.sequential(
            # input 1 x 4 x 4 array
            nn.Conv2d(1, 5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # state size 5 x 2 x 2
            nn.Conv2d(5, 10, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # state size 10 x 1 x 1
            nn.Linear(10, 4),
            nn.ReLU()
            # state size 4 x 1 x1
        )

    def forward(self, x):
        x = self.model(x)
        return torch.argmax(x) 