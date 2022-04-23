import config
import numpy as np
from torch.utils.data import Dataset

class TrainDataSet(Dataset):
    def __init__(self):
        self.x = np.load(config.DATASET_PATH+'x_train.npy')[:config.TRAIN_DATASET_SIZES]
        self.y = np.load(config.DATASET_PATH+'y_train.npy')[:config.TRAIN_DATASET_SIZES]
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

class TestDataSet(Dataset):
    def __init__(self):
        self.x = np.load(config.DATASET_PATH+'x_train.npy')[config.TRAIN_DATASET_SIZES:]
        self.y = np.load(config.DATASET_PATH+'y_train.npy')[config.TRAIN_DATASET_SIZES:]
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
