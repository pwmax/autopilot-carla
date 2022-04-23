import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TestDataSet
from model import Model

def eval_model(model, model_state):
    state = torch.load(model_state)
    model = model
    model.load_state_dict(state)
    model.eval()
    dataset = TestDataSet()
    dataloader = DataLoader(dataset, batch_size=64)
    loss_list = []
    criterion = nn.MSELoss()
   
    for x, y in dataloader:
        x, y = x.float(), y.float()
        out = model(x.reshape(x.size(0), 3, 66, 200))
        loss = criterion(out, y)
        loss_list.append(loss.item())

    print(sum(loss_list)/len(loss_list))

if __name__ == '__main__':
    eval_model(Model(), 'model_state.pth')