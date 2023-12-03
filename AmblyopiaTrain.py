# -*- coding: utf-8 -*-
from AmblyopiaDatasetPaper import *
from estcnn import *
from torch.utils.data.dataloader import DataLoader
import torch
import time
import os

dataset_path = r"E:\Dataset\amplyopia"
batch_size = 16
num_epoch = [10,20,30,50,500]
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_length = [250,500,1000,1500,2000,2500,3000]
dir_name = "./epoch" + str(num_epoch)
os.makedirs(dir_name,exist_ok=True)

for epochs in num_epoch:
    for length in data_length:
        dataset = AmblyopiaDataset(dataset_path, length)
        train_size = int(len(dataset) * 0.7)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        net = ESTCNN(16, length).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        text_name = dir_name + '/ESTCNN_result_' + str(length) + '.txt'

        with open(text_name, 'w') as f:
            for epoch in range(epochs):
                for step, (x, y) in enumerate(train_dataloader):
                    input = x.type(torch.cuda.FloatTensor).to(device)
                    label = y.to(device)
                    pred_y = net(input)
                    # print(label)
                    # pred_y = torch.argmax(net(input), dim=1)
                    # print(pred_y)
                    loss = criterion(pred_y, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if step % 10 == 0:
                        f.write("Epoch:{}, {}batch: loss={}\n".format(epoch, step, loss.item()))
                        print("Epoch:{}, {}个batch: loss={}".format(epoch, step, loss.item()))

                net.eval()
                with torch.no_grad():
                    test_loss, correct = 0, 0
                    for x, y in test_dataloader:
                        x = x.type(torch.cuda.FloatTensor).to(device)
                        y = y.to(device)
                        y_pred = net(x)
                        test_loss += criterion(y_pred, y).item()
                        correct += (y == torch.argmax(y_pred, axis=1)).sum().item()

                    test_loss /= len(test_dataloader)
                    correct /= len(test_dataset)
                    f.write("accuracy:{},average loss:{}\n".format(correct, test_loss))
                    f.write(time.strftime("%Y-%m-%d-%H:%M:%S\n", time.localtime()))
                    print("accuracy:{},average loss:{}".format(correct, test_loss))

        model_name = './epoch' + str(num_epoch) + '/ESTCNN_net_' + str(length) + '.pth'
        torch.save(net.state_dict(), model_name)

