import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys

sys.path.extend(['../'])
from feeders import tools
from feeders import vlms_tools



class Feeder_Shiftgcn_Pretrain(Dataset):
    def __init__(self, data_path, split, ntu_task='ntu60_xsub',zero_spilt_setting='ntu60_seen55_unseen5', label_path=None, p_interval=1, random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.ntu_task = ntu_task
        self.zero_spilt_setting = zero_spilt_setting
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        # split seen and unseen classes
        if self.zero_spilt_setting == 'ntu60_seen55_unseen5':
            unseen_classes = [10, 11, 19, 26, 56]   # ntu60_55/5_split
            seen_classes = list(set(range(60))-set(unseen_classes))  # ntu60
        elif self.zero_spilt_setting == 'ntu60_seen48_unseen12':
            unseen_classes = [3,5,9,12,15,40,42,47,51,56,58,59]  # ntu60_48/12_split
            seen_classes = list(set(range(60))-set(unseen_classes))
        elif self.zero_spilt_setting == 'ntu120_seen110_unseen10':
            unseen_classes = [4,13,37,43,49,65,88,95,99,106]  # ntu120_110/10_split
            seen_classes = list(set(range(120))-set(unseen_classes))
        elif self.zero_spilt_setting == 'ntu120_seen96_unseen24':
            unseen_classes = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]  # ntu120_96/24_split
            seen_classes = list(set(range(120))-set(unseen_classes))
        elif self.zero_spilt_setting == 'as_ntu60_seen55_unseen5_split1':
            unseen_classes = [4,19,31,47,51]   # ablation study split1
            seen_classes = list(set(range(60))-set(unseen_classes))
        elif self.zero_spilt_setting == 'as_ntu60_seen55_unseen5_split2':
            unseen_classes = [12,29,32,44,59]   # ablation study split2
            seen_classes = list(set(range(60))-set(unseen_classes))
        elif self.zero_spilt_setting == 'as_ntu60_seen55_unseen5_split3':
            unseen_classes = [7,20,28,39,58]   # ablation study split3
            seen_classes = list(set(range(60))-set(unseen_classes))
        else:
            raise NotImplementedError('Seen and unseen split errors!')
        label_dict = {}
        for idx, l in enumerate(seen_classes):
            label_dict[l] = idx
        # read data
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise NotImplementedError('Data & Label: data split only supports train/test')
        # delete
        unseen_samples_index_list = []
        for label_index, label_ele in enumerate(self.label):
            if label_ele in unseen_classes:
                unseen_samples_index_list.append(label_index)
        self.data = np.delete(self.data, unseen_samples_index_list, axis=0)
        self.label = np.delete(self.label, unseen_samples_index_list, axis=0)
        # refine label
        for label_index, label_ele in enumerate(self.label):
            self.label[label_index] = label_dict[label_ele.item()]
        # sample name
        if self.split == 'train':
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('Sample Name: data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = vlms_tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = vlms_tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)



if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)

