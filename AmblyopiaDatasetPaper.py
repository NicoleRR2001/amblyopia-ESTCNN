import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import os


class AmblyopiaDataset(Dataset):
    def __init__(self, data_dir, length):
        data_dir = os.path.join(data_dir, "amblyopia_processed_" + str(length))
        self.data_dir = data_dir
        self.label_dict = {'N': 0, 'R': 1, 'L': 2, 'B': 3}
        # Get eeg list
        self.eeg_list = glob.glob(data_dir + '\*')
        # Calculate len
        self.data_len = len(self.eeg_list)

    def __getitem__(self, index):
        filepath = self.eeg_list[index]
        filename = filepath.split('\\')[-1]
        data = np.load(filepath).transpose()
        data = np.expand_dims(data, axis=0)
        label = self.label_dict.get(filename.split('_')[1])
        return data, label

    def __len__(self):
        return self.data_len


"""print dataset and usage"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = r"E:\Dataset\amplyopia"
    dataset = AmblyopiaDataset(dataset_path, 500)
    print(dataset[0][1])
    # (1, 17, 2000)
    # print("Successfully.")
