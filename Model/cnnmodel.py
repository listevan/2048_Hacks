import torch
from torch import nn
from torch.nn import functional as F

# maybe try a CNN instead
class DQN(nn.Module):
    def __init__(self, nc):
        super(DQN, self).__init__()
        
        self.model = nn.Sequential(
            # input 1 x 4 x 4 array
            nn.Conv2d(nc, 5, kernel_size=7, padding='same', bias=False), # maybe conv3d
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(5, 20, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(20, 50, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 5 x 4 x 4
            nn.Conv2d(50, 100, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(100, 50, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(50, 10, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2, inplace=True),
            # state size  
            nn.Dropout(.3),
        )
        self.fc = nn.Linear(160, 4)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x