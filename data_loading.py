import h5py
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt
import pdb
import torch
import os
from torch.utils.data import Dataset
class freq_data(Dataset):
    # Constructor
    def __init__(self, path):
        file_freq = os.path.join(path, 'freq_norm.mat')
        file_rocof = os.path.join(path, 'rocof_norm.mat')
        freq_data, rocof_data = loading(file_freq, file_rocof)
        self.x, self.y, _, _ = separate_dataset(freq_data, rocof_data)
        self.len = self.x.shape[0]
       

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # Return the length
    def __len__(self):
        return self.len

def loading(file_freq, file_rocof):
     
    

    # loading total data
    file_f = h5py.File(file_freq, 'r')
    file_rocof = h5py.File(file_rocof, 'r')
    f_var = file_f.get('f')
    rocof_var = file_rocof.get('rf')
    f_var = np.array(f_var).T
    rocof_var = np.array(rocof_var).T
    return f_var, rocof_var

def separate_dataset(freq_data, rocof_data):
     

    total_dataset = np.hstack((freq_data[:,0:201],rocof_data[:,0:201], freq_data[:,-1:])) # here 201 is used just to
    # extract first 201 datapoints
    x = total_dataset[:,:-1] # contains freq and rocof datapoints
    y = total_dataset[:,-1] # contains inertia constant

    return x, y, freq_data[:,0:201],rocof_data[:,0:201]

if __name__ == '__main__':
    path= r"C:\Users\UW-User\Downloads\Projects\\"
    file_freq = path + 'freq_norm.mat'
    file_rocof = path + 'rocof_norm.mat'
    freq_data, rocof_data = loading(file_freq, file_rocof)
    _, _, f, rf = separate_dataset(freq_data, rocof_data)
    print("Number of data points for freq_norm:", f.shape[0])
    print("Number of data points for rocof_norm:", rf.shape[0])
    freq_norm.describe().T
    for i in range(f.shape[0]):
        plt.subplot(211)
        plt.plot(f[i,:])
        plt.subplot(212)
        plt.plot(rf[i,:])
    plt.show()