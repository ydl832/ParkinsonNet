# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from .ntu_read_skeleton import read_xyz
# visualization
import time
from scipy.signal import stft

# operation
from . import tools

small = False


class Feeder_PD(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 p_interval=1,
                 normalization=False,
                 mirroring=False,
                 debug=False,
                 mmap=True,
                 bone=False,
                 vel = True,
                 distance = False,
                 translation = False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.load_data(mmap)
        self.bone = bone
        self.vel = vel
        self.distance = distance
        self.translation = translation
        self.p_interval = p_interval
        if normalization:
            self.get_mean_map()
        self.mirroring = mirroring
    def load_data(self, mmap):
        # data: N C V T M  (100,3,125,21,1)
          
        # load label
        data = np.loadtxt(self.label_path, dtype=str)
        self.sample_name = data[:, 0]  
        self.label = data[:, 1].astype(int) 

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r+')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(
            axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def random_translation(self, ske_data):
        translate = np.eye(3)  
        random.random()
        t_x = random.uniform(-0.01, 0.01)  
        t_y = random.uniform(-0.01, 0.01)
        t_z = random.uniform(-0.01, 0.01)

        translate[0, 0] = translate[0, 0] + t_x
        translate[1, 1] = translate[1, 1] + t_y
        translate[2, 2] = translate[2, 2] + t_z

        data = np.dot(ske_data, translate)
        return data

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        index = index % len(self.data)
        
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        sample=self.sample_name[index]
#        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
#        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        C, T, V, M = data_numpy.shape
        # normalization
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        # processing
        if self.bone:
            ori_data = data_numpy
            for v1, v2 in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), 
                (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), 
                (14, 15),  (15, 16), (13, 17), (17, 18), (18, 19), (19, 20)):
                
                data_numpy[:, :, v2] = data_numpy[:, :, v2] - data_numpy[:, :, v1] 
        else:
            # Using the trajectory of joint 0
            trajectory = data_numpy[:, :, 0]
            # Shifting all joint coordinates so that joint 20 becomes the origin
            data_numpy = data_numpy - data_numpy[:, :, 0:1]
            # Restoring the original trajectory of joint 0
            data_numpy[:, :, 0] = trajectory
            
#        if self.translation:
#            data_numpy = np.transpose(data_numpy, (3, 1, 2, 0))
#            data_numpy = self.random_translation(data_numpy)
#            data_numpy = np.transpose(data_numpy, (3, 1, 2, 0))
            
        if self.distance:
            data_numpy_d = np.transpose(data_numpy, (3, 1, 2, 0))
            joint_4_coordinates = data_numpy_d[:, :, 4:5, :]
            #joint_0_coordinates = data_numpy_d[:, :, 0:1, :]
            distance4 = np.linalg.norm(data_numpy_d - joint_4_coordinates, axis=-1)
            #distance0 = np.linalg.norm(data_numpy_d - joint_0_coordinates, axis=-1)
            distance4 = distance4[..., np.newaxis]
            #distance0 = distance0[..., np.newaxis]

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
            velocity = data_numpy
        else:
            velocity = data_numpy[:, 1:] - data_numpy[:, :-1]
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            data_numpy[:, :-2] = acceleration
            data_numpy[:, -2:] = np.zeros_like(data_numpy[:, -2:])

#        
#        stft_features = np.zeros((C, T, V, M))
#        # ?????????
#        # ??????????STFT?????
#        for c in range(C):
#            for v in range(V):
#                signal = velocity[c, :, v].flatten()  # [125,1]
#                # ?????????(STFT)
#                f, t, Zxx = stft(signal, fs=25, window='hann', nperseg=13, noverlap=12)
#                # ??????
#                instantaneous_energy = np.sum(np.abs(Zxx)**2, axis=0)
#                stft_features[c, :len(instantaneous_energy), v, 0] = instantaneous_energy
#                
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.mirroring and bool(torch.bernoulli((0.5) * torch.ones(1))):
            data_numpy = tools.mirroring_v1(data_numpy)
        
        if self.distance:
           data_numpy = np.concatenate((data_numpy, distance4), axis=0)

        return data_numpy, label, sample

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        return tools.calculate_recall_precision(self.label, score)

    def is_training(self, state):
        self.state = state


def test(data_path, label_path, vid=None):
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        pose, = ax.plot(np.zeros(V * M), np.zeros(V * M), 'g^')
        ax.axis([-1, 1, -1, 1])

        for n in range(N):
            for t in range(T):
                x = data[n, 0, t, :, 0]
                y = data[n, 1, t, :, 0]
                z = data[n, 2, t, :, 0]
                pose.set_xdata(x)
                pose.set_ydata(y)
                fig.canvas.draw()
                plt.pause(1)


if __name__ == '__main__':
    data_path = "./data/NTU-RGB-D/xview/val_data.npy"
    label_path = "./data/NTU-RGB-D/xview/val_label.pkl"

    test(data_path, label_path, vid='S003C001P017R001A044')
