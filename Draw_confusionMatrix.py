from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
import torch
from estcnn import *
from AmblyopiaDatasetPaper import *
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os

data_length = [250, 500, 1000, 1500, 2000, 2500, 3000]
ratio = [0.5, 1, 2, 3, 4, 5, 6]
# epoch_list = [10, 20, 30, 50, 500]
epoch_list = [500]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_path = r"E:\Dataset\amplyopia"


# epoch = 500


def cm(cm, title, index):
    labels_name = ['N', 'R', 'L', 'B']
    proportion = []
    for i in cm:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(4, 4)
    pshow = np.array(pshow).reshape(4, 4)
    # print(pshow)
    config = {
        "font.family": 'Times New Roman',
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels_name))
    plt.xticks(tick_marks, labels_name, fontsize=12)
    plt.yticks(tick_marks, labels_name, fontsize=12)

    iters = np.reshape([[[i, j] for j in range(4)] for i in range(4)], (cm.size, 2))

    for i, j in iters:
        if i == j:
            plt.text(j, i - 0.12, format(cm[i][j]), va='center', ha='center', fontsize=12, color='white',
                     weight=5)
            plt.text(j, i + 0.12, pshow[i][j], va='center', ha='center', fontsize=12, color='white')
        else:
            plt.text(j, i - 0.12, format(cm[i][j]), va='center', ha='center', fontsize=12)
            plt.text(j, i + 0.12, pshow[i][j], va='center', ha='center', fontsize=12)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.savefig("{}.png".format(title))
    plt.close()


def roc(y_real, y_pred, num_classes, title):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    lw = 2
    dicts = {0: 'N', 1: 'R', 2: 'L', 3: 'B'}
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkcyan']
    for i in range(num_classes):
        label = (y_real != [i])
        label = label.astype(np.int)
        # print(label)
        y_score = y_pred[:, i]
        # y_score = np.abs(y_score - label)
        y_true = (y_real == [i])
        y_true = y_true.astype(np.int).tolist()
        y_score = y_score.tolist()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        # print(roc_auc)
        plt.plot(fpr, tpr, '--', color=colors[i], lw=lw,
                 label='ROC curve of Class %s (area = %0.2f)' % (dicts[i], roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(title)
    plt.close()


def window_size_load_model():
    for epoch in epoch_list:
        model_path = r"E:\pythonProject\EGG\Paper\amplyopia\epoch" + str(epoch) + "\ESTCNN_net_"
        for index, length in enumerate(data_length):
            model_name = model_path + str(length) + ".pth"
            net = ESTCNN(16, length)
            # load model
            dataset = AmblyopiaDataset(dataset_path, length)
            batch_size = 16
            correct = 0
            net.load_state_dict(torch.load(model_name))
            net = net.to(device)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
            net.eval()
            real_y = []
            pred_y = []
            score_y = []
            y_list = []
            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to(device).type(torch.cuda.FloatTensor)
                    y_list.extend(y.cpu().numpy().tolist())
                    y = y.to(device)
                    pred = net(x)
                    # pred = sm(pred)
                    pred_max = torch.max(pred, axis=1)
                    pred_y.extend(pred_max[1].cpu().numpy().tolist())
                    # score = pred[torch.nn.functional.one_hot(y,num_classes=4).type(torch.bool)]
                    score_y.extend(pred.cpu().numpy().tolist())
                    real_y.extend(y.cpu().numpy().tolist())
            # print(pred_y)
            # print(y_list)
            print(length)
            print(accuracy_score(pred_y, y_list))
            print(classification_report(pred_y, y_list))
            # print(score_y)
            # roc(real_y, score_y, 4,"{}-epoch ROC curve of {}-window size".format(epoch, length))
            # c = confusion_matrix(real_y, pred_y)
            # cm(c, "{}-epoch confusion matrix of {}-window_size".format(epoch, length), index)


def sequence_load_model():
    path = './pth'
    files = os.listdir(path)
    for file in files:
        model_path = os.path.join(path, file)
        data_set = AmblyopiaDataset(dataset_path, 2000)
        dataloader = DataLoader(data_set, 16)
        net = ESTCNN(16, 2000).to(device)
        net.load_state_dict(torch.load(model_path))
        net.eval()
        pred_y = []
        list_y = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device).type(torch.cuda.FloatTensor)
                list_y.extend(y.numpy().tolist())
                pred = net(x)
                pred = torch.max(pred, axis=1)
                pred_y.extend(pred[1].cpu().numpy().tolist())
        print(accuracy_score(pred_y, list_y))
        print(classification_report(pred_y, list_y))


if __name__ == '__main__':
    sequence_load_model()
