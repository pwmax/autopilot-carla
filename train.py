import torch
import torch.nn as nn
import keyboard
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import TrainDataSet
from model import Model

batch_size = 32
learning_rate = 0.01
num_epoch = 1000
loss_list = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = TrainDataSet()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Model()
model.train()
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

def save_model(epoch):
    if keyboard.is_pressed('NUM_3'):
        torch.save(model.state_dict(), f'/autopilot-carla/{epoch}model.pth')

def show_loss():
    if keyboard.is_pressed('NUM_4'):
        plt.ylim(0, 0.090)
        plt.plot(loss_list)
        plt.show()

def main():
    for epoch in range(num_epoch):
        for data, target in dataloader:
            
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            out = model(data.reshape(data.size(0), 3, 66, 200))

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            print(loss.item(), epoch)

            save_model(epoch)
            show_loss()

if __name__ == '__main__':
    main()