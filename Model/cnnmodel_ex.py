import torch
from torch import nn
from torch.nn import functional as F

# maybe try a CNN instead
class DQN(nn.Module):
    def __init__(self, nc):
        super(DQN, self).__init__()
        
        self.model = nn.Sequential(
            # input 11 x 4 x 4 array
            nn.Conv3d(nc, 5, kernel_size=7, padding='same', bias=False), # maybe conv3d
            nn.BatchNorm3d(5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(5, 20, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm3d(20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(20, 50, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm3d(50),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 5 x 4 x 4
            nn.Conv3d(50, 20, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm3d(20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(20, 10, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm3d(10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(10, 5, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm3d(5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size  
            nn.Dropout(.3),
        )
        self.fc = nn.Linear(880, 4)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x