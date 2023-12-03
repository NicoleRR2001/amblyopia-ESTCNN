import torch
from estcnn import *
from AmblyopiaDatasetPaper import *
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import os

data_length = [250,500,1000,1500,2000,2500,3000]
ratio = [0.5,1,2,3,4,5,6]
epoch = 20
dataset_path = r"E:\Dataset\amplyopia"
model_path = r"E:\pythonProject\EGG\Paper\amplyopia\epoch" + str(epoch) + "\ESTCNN_net_"

result_dir = "./epoch" + str(epoch) + "/result"
os.makedirs(result_dir,exist_ok=True)
with open(result_dir+"/result.txt","w") as f:
    plt.title("Amblyopia Dataset in "+str(epoch) + " epochs")
    plt.ylim([97, 100])
    plt.xlabel("ratio of sample rate")
    plt.ylabel("accuracy")

    for index, length in enumerate(data_length):
        model_name = model_path + str(length) + ".pth"
        net = ESTCNN(16, length)
        dataset_path = r"E:\Dataset\amplyopia"
        dataset = AmblyopiaDataset(dataset_path, length)
        batch_size = 16
        correct = 0
        net.load_state_dict(torch.load(model_name))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
        net.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device).type(torch.cuda.FloatTensor)
                y = y.to(device)
                pred_y = torch.argmax(net(x), axis=1)
                correct += (pred_y == y).sum().item()

        accuracy = round(100 * correct / len(dataset), 2)
        print(accuracy)
        f.write(str(accuracy) + "\n")

        plt.scatter(ratio[index], accuracy, c='b')

    plt.savefig(result_dir+"/result.png")
    plt.show()