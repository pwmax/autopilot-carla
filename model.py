import torch 
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 54, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(54, 64, 3, stride=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = x.view(x.size(0), -1)
        x = self.block2(x)
        return x