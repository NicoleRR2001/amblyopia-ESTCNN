import matplotlib.pyplot as plt
import numpy as np

epoch_list = [10,20,30,50,500]

ratio = [0.5,1,2,3,4,5,6]

max_list = []
index_list = []

for epoch in epoch_list:
    res_path = './epoch' + str(epoch) + '/result/result.txt'
    with open(res_path,'r') as f:
        data_list = f.read().split('\n')[:-1]
        y = []
        for data in data_list:
            y.append(float(data))
        index_list.append(np.argmax(y))
        max_list.append(max(y))
        plt.plot(ratio,y,'-*',label=str(epoch)+" epochs")
        l = list(map(str, y))
        # print(l)
        s = ' & '.join(l)
        print(' & ' + s + '\\\\')

plt.title("Performance in Different During Time and Epochs")
plt.ylim([90, 100])
plt.xlabel("Signal Duration / seconds")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
print(index_list)
print(max_list)