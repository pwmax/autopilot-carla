import torch 
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=54, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=54, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(in_features=1152, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = x.view(x.size(0), -1)
        x = self.block2(x)
        return x