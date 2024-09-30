import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import clip
from PIL import Image
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction
from tools import *
# from KLLoss import KLLoss
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import manifold
import numpy as np
from scipy.special import binom

unseen_classes = [10, 11, 19, 26, 56]   # ntu60_55/5_split
# unseen_classes = [4,19,31,47,51]   # ablation study ntu60 split1
# unseen_classes = [12,29,32,44,59]   # ablation study ntu60 split2
# unseen_classes = [7,20,28,39,58]   # ablation study ntu60 split3
# unseen_classes = [3,5,9,12,15,40,42,47,51,56,58,59]  # ntu60_48/12_split
# unseen_classes = [4,13,37,43,49,65,88,95,99,106]  # ntu120_110/10_split
# unseen_classes = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]  # ntu120_96/24_split
seen_classes = list(set(range(60))-set(unseen_classes))  # ntu60
train_label_dict = {}
train_label_map_dict = {}
for idx, l in enumerate(seen_classes):
    tmp = [0] * len(seen_classes)
    tmp[idx] = 1
    train_label_dict[l] = tmp
    train_label_map_dict[l] = idx
test_zsl_label_dict = {}
test_zsl_label_map_dict = {}
for idx, l in enumerate(unseen_classes):
    tmp = [0] * len(unseen_classes)
    tmp[idx] = 1
    test_zsl_label_dict[l] = tmp
    test_zsl_label_map_dict[l] = idx
test_gzsl_label_dict = {}
test_gzsl_label_map_dict = {}
for idx, l in enumerate(range(60)):
    tmp = [0] * 60
    tmp[idx] = 1
    test_gzsl_label_dict[l] = tmp
    test_gzsl_label_map_dict[l] = idx


scaler = torch.cuda.amp.GradScaler()

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # extract attributes feature and action descriptions features
        self.label_semantics = torch.load('/root/autodl-tmp/FGE/semantics/ntu120_label_semantic_feature.tar')
        print('Extract CLIP Semantics Successful!')
        # load model
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        # load skeleton action recognition model
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
        

        
    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        # print(self.model)
        # loss function setting
        self.loss_ce = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            # weights = OrderedDict([["feature_extractor."+k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        print("Load model done.")

        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                filter(lambda p:p.requires_grad, self.model.parameters()),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        
    
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
            print("Load train data done.")
        self.data_loader['test_zsl'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_zsl_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        print("Load zsl test data done.")
        self.data_loader['test_gzsl'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_gzsl_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        print("Load gzsl test data done.")

    
    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()
    
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time
    
    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        pred_list = []
        label_list = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        # semantics
        label_semantics_seen = self.label_semantics[seen_classes].cuda(self.output_device)
        # label_semantics_unseen = self.label_semantics[unseen_classes].cuda(self.output_device)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                b,_,_,_,_ = data.size()
            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()

            label_reindex_seen = torch.tensor([train_label_map_dict[l.item()] for l in label]).long().cuda(self.output_device)
            # label_reindex = torch.tensor([test_gzsl_label_map_dict[l.item()] for l in label]).cuda(self.output_device)
            # print(label[:20], label_reindex_seen[:20])
            output = self.model(data, label_semantics_seen)
            loss = self.loss_ce(output, label_reindex_seen)

            loss.backward()
            self.optimizer.step()
            # scaler.scale(loss).backward()

            # scaler.step(self.optimizer)
            # scaler.update()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            pred_list.append(predict_label.data.cpu().numpy())
            label_list.append(label_reindex_seen.data.cpu().numpy())
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)
        # print(label_list[:20], pred_list[:20])
        acc = np.mean((pred_list==label_list))
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), acc*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))



        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')


    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        # semantics
        label_semantics_seen = self.label_semantics[seen_classes].cuda(self.output_device)
        label_semantics_unseen = self.label_semantics[unseen_classes].cuda(self.output_device)
        for ln in loader_name:
            pred_list = []
            label_list = []
            loss_value = []
            score_frag = []
            feature_list = []
            sim_matrix_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)

            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    # print(data.size())
                    b, _, _, _, _ = data.size()
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    if ln == "test_zsl":
                        label_semantics_unseen = self.label_semantics[unseen_classes].cuda(self.output_device)
                        label_reindex_unseen = torch.tensor([test_zsl_label_map_dict[l.item()] for l in label]).cuda(self.output_device)
                        output, feature = self.model(data, label_semantics_unseen)
                        feature_list.append(feature.cpu().numpy())
                        value, predict_label = torch.max(output.data, 1)
                        pred_list.append(predict_label.data.cpu().numpy())
                        label_list.append(label_reindex_unseen.data.cpu().numpy())
                    if ln == "test_gzsl":
                        label_semantics_unseen = self.label_semantics[:60,:].cuda(self.output_device)
                        label_reindex_unseen = torch.tensor([test_gzsl_label_map_dict[l.item()] for l in label]).cuda(self.output_device)
                        output = self.model(data, label_semantics_unseen)
                        output = F.softmax(output, 1)
                        sim_matrix_list.append(output)
                        label_list.append(label_reindex_unseen.data.cpu().numpy())
                    step += 1

            if ln == 'test_zsl':
                label_list= np.concatenate(label_list)
                pred_list= np.concatenate(pred_list)
                feature_list = np.concatenate(feature_list, axis=0)
                acc = np.mean(pred_list==label_list)
                self.print_log('\tTop{}: {:.2f}%'.format(1, acc*100))
                # acc for each class:
                confusion = confusion_matrix(label_list, pred_list)
                list_diag = np.diag(confusion)
                list_raw_sum = np.sum(confusion, axis=1)
                each_acc = list_diag / list_raw_sum
                with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(each_acc)
                    writer.writerows(confusion)
            if ln == 'test_gzsl':
                label_list= np.concatenate(label_list)
                sim_matrix = torch.cat(sim_matrix_list,dim=0).cuda(self.output_device)
                calibration_factor_list = [i/100 for i in range(5, 100, 5)]
                result = []
                for calibration_factor in calibration_factor_list:
                    sim_matrix_loop = sim_matrix.clone()
                    tmp = torch.zeros_like(sim_matrix_loop)
                    tmp[:, seen_classes] = calibration_factor
                    sim_matrix_loop = sim_matrix_loop - tmp
                    sim_matrix_pred_idx = torch.max(sim_matrix_loop, dim=1)[1]
                    sim_matrix_pred_idx = sim_matrix_pred_idx.data.cpu().numpy()
                    acc_seen = []
                    acc_unseen = []
                    for tl, pl in zip(label_list, sim_matrix_pred_idx):
                        if tl in seen_classes:
                            acc_seen.append(int(tl)==int(pl))
                        else:
                            acc_unseen.append(int(tl)==int(pl))
                    acc_seen = sum(acc_seen) / len(acc_seen)
                    acc_unseen = sum(acc_unseen) / len(acc_unseen)
                    result.append((calibration_factor, acc_unseen, acc_seen))

                # best_calibration_factor = -1
                # best_accuracy_unseen = -1
                # best_accuracy_seen = -1
                # best_harmonic_mean_acc = -1
                for calibration_factor, accuracy_unseen, accuracy_seen in result:
                    harmonic_mean_acc = 2*accuracy_seen*accuracy_unseen/(accuracy_seen+accuracy_unseen)
                    self.print_log('\tCalibration Factor: {:.8f}'.format(calibration_factor))
                    self.print_log('\tSeen Acc: {:.2f}%'.format(accuracy_seen*100))
                    self.print_log('\tUnseen Acc: {:.2f}%'.format(accuracy_unseen*100))
                    self.print_log('\tHarmonic Mean Acc: {:.2f}%'.format(harmonic_mean_acc*100))
                    # if harmonic_mean_acc > best_harmonic_mean_acc:
                    #     best_harmonic_mean_acc = harmonic_mean_acc
                    #     best_accuracy_unseen = accuracy_unseen
                    #     best_accuracy_seen = accuracy_seen
                    #     best_calibration_factor = calibration_factor
        return feature_list, label_list

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                # self.train(epoch, save_model=save_model)
                # self.train(epoch, save_model=True)
                # zsl
                feature, label = self.eval(epoch, save_score=self.arg.save_score, loader_name=['test_zsl'])
                np.savez('/root/autodl-tmp/FGE/visualization/unseen_match_temporal_label.npz', x_test=feature, y_test=label)
                break
                # gzsl
                # self.eval(epoch, save_score=self.arg.save_score, loader_name=['test_gzsl'])

            # test the best model
            # print(glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*')))
            # print(glob.glob(os.path.join(self.arg.work_dir, 'runs/'+str(self.best_acc_epoch)+'*')))
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
    




def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='LLMs for Action Recognition')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for stroing results.')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config/nturgbd-cross-view/default.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=32, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-zsl-args', action=DictAction, default=dict(), help='the arguments of data loader for test zsl')
    parser.add_argument('--test-feeder-gzsl-args', action=DictAction, default=dict(), help='the arguments of data loader for test gzsl')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--text_weights', default=None, help='the weights for network initialization')
    parser.add_argument('--rgb_weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-alpha1', type=float, default=0.8)
    parser.add_argument('--loss-alpha2', type=float, default=0.8)
    parser.add_argument('--loss-alpha3', type=float, default=0.8)
    parser.add_argument('--te-lr-ratio', type=float, default=1)

    return parser


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

