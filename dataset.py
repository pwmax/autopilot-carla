import numpy as np
from torch.utils.data import Dataset

class TrainDataSet(Dataset):
    def __init__(self):
        self.x = np.load('/autopilot-carla/x_train.npy')
        self.y = np.load('/autopilot-carla/y_train.npy')
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])