import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

import sys
sys.path.append("./model/Temporal_shift/")

from cuda.shift import Shift
import copy


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(25*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        index_array = np.empty(25*in_channels).astype(np.int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(25*out_channels).astype(np.int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous()

        # shift1
        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)
        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        cube_feature = x.view(N, M, c_new, -1, V)
        cube_feature = cube_feature.mean(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x), x, cube_feature




class ModelMatch(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(ModelMatch, self).__init__()
        # pretraining model
        self.feature_extractor = Model(num_class, num_point, num_person, graph, graph_args, in_channels)
        for p in self.parameters():
            p.requires_grad = False
        self.spatial_prototype = nn.Embedding(80, 256)  # 256 25*4
        self.temporal_prototype = nn.Embedding(80, 256)  # 256 16*3
        self.relu = nn.ReLU()

        self.fc_spatial = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.spatial_project = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU()
        )
        self.update_spatial = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])

        self.fc_temporal = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.temporal_project = nn.Sequential(
            nn.Linear(256, 768),
            nn.ReLU()
        )
        self.update_temporal = nn.ModuleList([copy.deepcopy(self.fc_temporal) for i in range(3)])
        self.memory_temporal = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])
        self.recall_temporal = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])

    def forward(self, x, spatial_round, temporal_round):
        _, pooling_feature, cube_feature = self.feature_extractor(x)  # n, 256    n, 256, 16, 25
        b, _, _, _ = cube_feature.size()
        # prototype
        sp = self.spatial_prototype.weight.unsqueeze(0).expand(b, -1, -1).permute(0,2,1)
        sp_list = []
        for idx, dec in enumerate(self.update_spatial):
            sp_select = torch.einsum('ncj,ncv->njv', cube_feature.mean(2), sp)  # n, 25, 100(a)
            correlation_value_spatial = F.softmax(sp_select, dim=2)  # n, 25, 100(a)
            _, _, att_num = correlation_value_spatial.size()
            pro_indices = correlation_value_spatial.topk(80, dim=2)[1][:,:,80-(idx+1)*20:]  # 20
            batch_indices = torch.arange(b).unsqueeze(1).unsqueeze(2)
            joint_indices = torch.arange(25).unsqueeze(0).unsqueeze(2)
            corr = F.softmax(sp_select, dim=1)
            mask = torch.ones((b,25,80)).cuda()
            mask[batch_indices,joint_indices,pro_indices] = 1e-12
            corr = corr * mask
            sp = torch.einsum('ncj,njv->nvc', cube_feature.mean(2), corr) # n 75(a)  256
            sp = dec(sp).permute(0,2,1)  # n 256 75(a)
            sp_list.append(sp.mean(2))
        sp_proj_list = []
        for sp_ele in sp_list:
            sp_proj_list.append(F.normalize(self.spatial_project(sp_ele), p=2, dim=1))
        # prototype
        tp = self.temporal_prototype.weight.unsqueeze(0).expand(b, -1, -1).permute(0,2,1)  
        temporal_feature = cube_feature.mean(3)
        crop_feature = temporal_feature
        crop_feature_list = [temporal_feature[:,:,:5], temporal_feature[:,:,:10], temporal_feature]
        tp_list = []
        tp_attention_list = []
        for idx, dec in enumerate(self.update_temporal):
            crop_feature = crop_feature_list[idx]
            tp_select = torch.einsum('nct,nca->nta', crop_feature, tp)  # n frame_num 64(a)
            correlation_value_temporal = F.softmax(tp_select, dim=1)  # n, frame_num, 64(a)
            _, frame_num, att_num = correlation_value_temporal.size()
            tp_attention_list.append(tp_select)
            tp_cur = torch.einsum('nct,nta->nac', crop_feature, correlation_value_temporal) # n frame_num  256
            tp_recall = F.sigmoid(self.recall_temporal[idx](tp_cur).permute(0,2,1)) * tp
            tp_remember = F.sigmoid(self.memory_temporal[idx](tp_cur).permute(0,2,1)) * dec(tp_cur).permute(0,2,1)
            tp = tp_recall + tp_remember
            tp_list.append(tp.mean(2))
        tp_proj_list = []
        for tp_ele in tp_list:
            tp_proj_list.append(F.normalize(self.temporal_project(tp_ele), p=2, dim=1))

        # semantic 
        spatial_fg_norm = F.normalize(spatial_round[2], p=2, dim=-1)   # 55(5) 10 768
        spatial_mg_norm = F.normalize(spatial_round[1], p=2, dim=-1)   # 55(5) 10 768
        spatial_cg_norm = F.normalize(spatial_round[0], p=2, dim=-1)   # 55(5) 10 768
        spatial_sem_norm_list = [spatial_cg_norm, spatial_mg_norm, spatial_fg_norm]
        temporal_fg_norm = F.normalize(temporal_round[2], p=2, dim=-1)   # 55(5) 10 768
        temporal_mg_norm = F.normalize(temporal_round[1], p=2, dim=-1)   # 55(5) 10 768
        temporal_cg_norm = F.normalize(temporal_round[0], p=2, dim=-1)   # 55(5) 768
        temporal_sem_norm_list = [temporal_cg_norm, temporal_mg_norm, temporal_fg_norm]
        # multiply
        logits_spatial_list = []
        logits_spatial_cg = torch.einsum('nd,ckd->nck', sp_proj_list[0], spatial_cg_norm).topk(10, dim=2)[0].mean(2)  # top3
        logits_spatial_mg = torch.einsum('nd,ckd->nck', sp_proj_list[1], spatial_mg_norm).topk(10, dim=2)[0].mean(2)  # n 55 10
        logits_spatial_fg = torch.einsum('nd,ckd->nck', sp_proj_list[2], spatial_fg_norm).topk(10, dim=2)[0].mean(2)
        logits_spatial_list.append(logits_spatial_cg*0.1)  # n 55  ntu60-0.1
        logits_spatial_list.append(logits_spatial_mg*0.1)  # n 55
        logits_spatial_list.append(logits_spatial_fg*0.1)  # n 55
        logits_temporal_list = []
        logits_temporal_cg = torch.einsum('nd,ckd->nck', tp_proj_list[0], temporal_cg_norm).topk(10, dim=2)[0].mean(2)
        logits_temporal_mg = torch.einsum('nd,ckd->nck', tp_proj_list[1], temporal_mg_norm).topk(10, dim=2)[0].mean(2) # n 55 10
        logits_temporal_fg = torch.einsum('nd,ckd->nck', tp_proj_list[2], temporal_fg_norm).topk(10, dim=2)[0].mean(2)
        logits_temporal_list.append(logits_temporal_cg*0.1)  # n 55
        logits_temporal_list.append(logits_temporal_mg*0.1)  # n 55
        logits_temporal_list.append(logits_temporal_fg*0.1)  # n 55
        
        return logits_spatial_list, logits_temporal_list

    


        



