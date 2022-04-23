import torch
import torch.nn as nn
import keyboard
import matplotlib.pyplot as plt
import config
from torch.utils.data import DataLoader
from dataset import TrainDataSet
from model import Model

class Trainer:
    def __init__(self, model, train_dataset, model_state=None):
        self.model = model
        self.model_state = model_state
        self.train_dataset = train_dataset
        self.loss_list = []
        self.main()
    
    def main(self):
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)

        model = self.model
        loss_list = self.loss_list
        model.to(config.DEVICE)
        if self.model_state:
            state = torch.load(self.model_state)
            model.load_state_dict(state)
        model.train()

        dataloader = DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()

        for epoch in range(config.NUM_EPOCH):
            for x, y in dataloader:
                
                x, y = x.to(config.DEVICE).float(), y.to(config.DEVICE).float()
                x = x.reshape(x.size(0), 3, 66, 200)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

                self._show_loss(loss_list, '3')
                print('epoc[%i/%i] loss=%.5f' % (epoch, config.NUM_EPOCH, loss.item()))

        torch.save(model.state_dict(), config.MODEL_PATH+'model_state.pth')

    def _show_loss(self, loss_list, key='3'):
        if keyboard.is_pressed(key):
            plt.plot(loss_list)
            plt.ylim(0, 0.1)
            plt.show()

if __name__ == '__main__':
    Trainer(Model(), TrainDataSet(), model_state='model_state.pth')