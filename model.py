import torch 
import torch.nn as nn
from torchsummary import summary

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, maxpool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU()
        self.maxpool = maxpool
        self.max_pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.max_pool(x)
        return x

class Model(nn.Module):
    def __init__(self, in_channels=3, out_size=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            Block(in_channels, 16),
            Block(16, 16, True),

            Block(16, 32),
            Block(32, 32, True),

            Block(32, 64),
            Block(64, 64, True),

            Block(64, 128),
            Block(128, 128, True),

            Block(128, 128),
            Block(128, 128, True),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(1536, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, out_size)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

if __name__ == '__main__':
    model = Model()
    summary(model, input_size=(3, 66, 200), device='cpu')
    data = torch.ones(1, 3, 66, 200)
    out = model(data)
    print(out.shape)