import torch
import numpy as np


class Feeder_Shiftgcn_Pretrain(torch.utils.data.Dataset):

    def __init__(self, data_path, split, pku_task='pkuv1_xsub',zero_spilt_setting='pkuv1_seen46_unseen5',label_path=None, p_interval=1, random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):

        self.data_path = data_path
        self.pku_task = pku_task
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
        

    def load_data(self):
        npz_data = np.load(self.data_path)
        # read all training samples
        # self.data = npz_data['x_train']
        # self.label = npz_data['y_train'].squeeze(1)
        # split seen and unseen classes
        if self.zero_spilt_setting == 'pkuv1_seen46_unseen5':
            self.unseen_classes = [1, 9, 20, 34, 50]   # pkuv1_seen46_unseen5
            seen_classes = list(set(range(51))-set(self.unseen_classes))  # pkuv1
        elif self.zero_spilt_setting == 'pkuv1_seen39_unseen12':
            self.unseen_classes = [3,7,11,15,19,21,25,31,33,36,43,48]  # npkuv1_seen39_unseen12
            seen_classes = list(set(range(51))-set(self.unseen_classes))
        elif self.zero_spilt_setting == 'pkuv2_seen46_unseen5':
            self.unseen_classes = [1, 9, 20, 34, 50]   # pkuv2_seen46_unseen5
            seen_classes = list(set(range(51))-set(self.unseen_classes))
        elif self.zero_spilt_setting == 'pkuv2_seen39_unseen12':
            self.unseen_classes = [3,7,11,15,19,21,25,31,33,36,43,48]  # npkuv2_seen39_unseen12
            seen_classes = list(set(range(51))-set(self.unseen_classes))
        elif self.zero_spilt_setting == 'as_pkuv1_seen46_unseen5_split1':
            self.unseen_classes = [3,14,29,31,49]  # ablation study split1
            seen_classes = list(set(range(51))-set(self.unseen_classes))
        elif self.zero_spilt_setting == 'as_pkuv1_seen46_unseen5_split2':
            self.unseen_classes = [2,15,39,41,43]  # ablation study split2
            seen_classes = list(set(range(51))-set(self.unseen_classes))
        elif self.zero_spilt_setting == 'as_pkuv1_seen46_unseen5_split3':
            self.unseen_classes = [4,12,16,22,36]  # ablation study split3
            seen_classes = list(set(range(51))-set(self.unseen_classes))
        else:
            raise NotImplementedError('Seen and unseen split errors!')
        label_dict = {}
        for idx, l in enumerate(seen_classes):
            label_dict[l] = idx
        # read data
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train'].squeeze(1)
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = npz_data['y_test'].squeeze(1)
        else:
            raise NotImplementedError('Data & Label: data split only supports train/test')
        unseen_samples_index_list = []
        for label_index, label_ele in enumerate(self.label):
            if label_ele in self.unseen_classes:
                unseen_samples_index_list.append(label_index)
        self.data = np.delete(self.data, unseen_samples_index_list, axis=0)
        self.label = np.delete(self.label, unseen_samples_index_list, axis=0)
        # refine label
        # print(self.label)
        for label_index, label_ele in enumerate(self.label):
            self.label[label_index] = label_dict[label_ele.item()]
        # print(self.label)
        # sample name
        if self.split == 'train':
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('Sample Name: data split only supports train/test')
        

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> tuple:

        data = self.data[index]
        label = self.label[index]

        return data, label, index
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
