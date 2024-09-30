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

        # self.spatial_prototype = nn.Embedding(256, 100)  # 256 25*4
        self.spatial_prototype = nn.Embedding(160, 256)  # 256 25*4
        # self.spatial_prototype = nn.Parameter(torch.ones(256, 100), requires_grad=True)  # 256 25*4
        # self.spatial_group = nn.Embedding(25, 5)
        self.temporal_prototype = nn.Embedding(80, 256)  # 256 16*3
        # self.temporal_prototype = nn.Parameter(torch.ones(256, 100))  # 256 25*4
        # self.temporal_memory = nn.ModuleList([copy.deepcopy(self.temporal_prototype) for i in range(3)])
        # self.multihead_attn = nn.MultiheadAttention(256, 8, batch_first=True)

        # self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        # self.norm_layer = nn.LayerNorm(256)

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
        self.memory_spatial = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])
        self.forget_spatial = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])
        # self.output_spatial = nn.ModuleList([copy.deepcopy(self.fc_spatial) for i in range(3)])
        self.proj_spatials = nn.ModuleList([copy.deepcopy(self.spatial_project) for i in range(3)])

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
        self.proj_temporals = nn.ModuleList([copy.deepcopy(self.temporal_project) for i in range(3)])
        # self.tem_group = nn.Embedding(3, 256)
        # self.tem_group = nn.Parameter(torch.randn(3, 16), requires_grad=True)

        self.fc_spatial_temporal = nn.Linear(768*2, 768)

        # self.logit_scale_spatial = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # contrastive learning
        # self.avg_proptype = torch.randn(3, 256, 55)
        # self.avg_proj = nn.ModuleList([copy.deepcopy(self.spatial_project) for i in range(3)])

    def forward(self, x, spatial_round, temporal_round, label_semantics):
        _, pooling_feature, cube_feature = self.feature_extractor(x)  # n, 256    n, 256, 16, 25
        b, _, _, _ = cube_feature.size()
        # prototype
        sp = self.spatial_prototype.weight.unsqueeze(0).expand(b, -1, -1).permute(0,2,1)
        # sp = self.spatial_prototype.weight.unsqueeze(0).expand(b,-1,-1)  # 100 n 256
        # sp = self.spatial_prototype.unsqueeze(0).expand(b, -1, -1)
        # spatial instrcution
        sp_list = []
        for idx, dec in enumerate(self.update_spatial):
            sp_select = torch.einsum('ncj,ncv->njv', cube_feature.mean(2), sp)  # n, 25, 100(a)
            

            correlation_value_spatial = F.softmax(sp_select, dim=2)  # n, 25, 100(a)
            _, _, att_num = correlation_value_spatial.size()
            pro_indices = correlation_value_spatial.topk(160, dim=2)[1][:,:,160-(idx+1)*20:]
            batch_indices = torch.arange(b).unsqueeze(1).unsqueeze(2)
            joint_indices = torch.arange(25).unsqueeze(0).unsqueeze(2)
            corr = F.softmax(sp_select, dim=1)
            mask = torch.ones((b,25,160)).cuda()
            mask[batch_indices,joint_indices,pro_indices] = 1e-12
            corr = corr * mask
            
            # sp = self.multihead_attn(sp, cube_feature.mean(2).permute(0,2,1), cube_feature.mean(2).permute(0,2,1))[0]
            # sp = torch.einsum('ncj,njv->nvc', cube_feature.mean(2), correlation_value_spatial.topk(att_num-25, dim=2)[0]) # n 75(a)  256
            sp = torch.einsum('ncj,njv->nvc', cube_feature.mean(2), corr) # n 75(a)  256
            sp = dec(sp).permute(0,2,1)  # n 256 75(a)
            sp_list.append(sp.mean(2))
            # sp = dec(sp)  # n  75(a)  256
            # sp_list.append(sp.mean(1))
        sp_proj_list = []
        # for sp_ele, proj in zip(sp_list, self.proj_spatials):
        #     sp_proj_list.append(F.normalize(proj(sp_ele), p=2, dim=1))
        for sp_ele in sp_list:
            sp_proj_list.append(F.normalize(self.spatial_project(sp_ele), p=2, dim=1))
        # sp = sp + cube_feature.mean(2)
        # prototype
        tp = self.temporal_prototype.weight.unsqueeze(0).expand(b, -1, -1).permute(0,2,1)  
        
        # tem_group = self.tem_group.weight
        # print(tem_group)
        # _, tem_group_idx = torch.max(tem_group, dim=0)
        # tem_group_idx = F.one_hot(tem_group_idx, num_classes=3).T   # 3 16
        # add
        # tem_group[1,:] += tem_group[0,:]
        # tem_group[2,:] += tem_group[1,:]
        # print(tem_group)
        temporal_feature = cube_feature.mean(3)
        crop_feature = temporal_feature
        # group_feature = torch.einsum('nct,gt->ngc', cube_feature.mean(3), tem_group)  # n 3 256
        # temporal instruction
        crop_feature_list = [temporal_feature[:,:,:5], temporal_feature[:,:,:10], temporal_feature]
        tp_list = []
        tp_attention_list = []
        for idx, dec in enumerate(self.update_temporal):
            # crop_feature = temporal_feature[:,:,tem_group[idx]==1]  # n 256 frame_num
            # mask = tem_group[idx].view(1, -1).unsqueeze(0).expand(b,256,-1)
            # time_gate = F.sigmoid(torch.einsum('nct,gc->ntg',temporal_feature,tem_group[idx].view(1, -1)))
            # crop_feature = temporal_feature * time_gate.permute(0,2,1).expand(-1, 256, -1)
            crop_feature = crop_feature_list[idx]
            # memeory = self.temporal_memory[idx].weight.unsqueeze(0).expand(b, -1, -1)
            # crop_feature = temporal_feature
            tp_select = torch.einsum('nct,nca->nta', crop_feature, tp)  # n frame_num 64(a)
            correlation_value_temporal = F.softmax(tp_select, dim=1)  # n, frame_num, 64(a)
            _, frame_num, att_num = correlation_value_temporal.size()
            tp_attention_list.append(tp_select)

            # time_indices = correlation_vaSlue_temporal.topk(5, dim=1)[1]
            # batch_indices = torch.arange(b).unsqueeze(1).unsqueeze(2)
            # pro_indices = torch.arange(256).unsqueeze(0).unsqueeze(2)

            # crop_new = crop_feature[batch_indices,pro_indices,time_indices]
            # print(corr.size())
            # corr = F.softmax(torch.einsum('nct,nca->nta', crop_new, tp),dim=1)
            tp_cur = torch.einsum('nct,nta->nac', crop_feature, correlation_value_temporal) # n frame_num  256
            # tp_cur = F.sigmoid(tp_cur) * tp.permute(0,2,1) + tp_cur
            # tp = dec(tp_cur).permute(0,2,1)
            # memory
            # forget
            tp_forget = F.sigmoid(self.forget_spatial[idx](tp_cur).permute(0,2,1)) * tp
            tp_remember = F.sigmoid(self.memory_spatial[idx](tp_cur).permute(0,2,1)) * dec(tp_cur).permute(0,2,1)
            tp = tp_forget + tp_remember
            # tp = dec(tp_cur).permute(0,2,1)

            # tp = F.sigmoid(self.forget_spatial[idx](tp_cur).permute(0,2,1)) * tp + dec(tp_cur).permute(0,2,1)


            # tp_cur_r = self.memory_spatial[idx](tp_cur).permute(0,2,1)
            # tp_forget = self.forget_spatial[idx](tp_cur).permute(0,2,1)
            # tp_ct = F.sigmoid(tp_cur_p)*F.tanh(tp_cur_r)+ F.sigmoid(tp_forget)*tp
            # tp_cur_o = self.output_spatial[idx](tp_cur).permute(0,2,1)
            # tp = tp_cur_o*F.tanh(tp_ct)
            # tp_list.append(tp_cur.mean(1))
            tp_list.append(tp.mean(2))
        tp_proj_list = []
        # for tp_ele, proj in zip(tp_list, self.proj_temporals):
        #     tp_proj_list.append(F.normalize(proj(tp_ele), p=2, dim=1))
        for tp_ele in tp_list:
            tp_proj_list.append(F.normalize(self.temporal_project(tp_ele), p=2, dim=1))
        # for idx, dec in enumerate(self.update_temporal):
        #     tp = torch.einsum('nck,nct->nkt', cube_feature.mean(3), tp)  # n, 16, 16
        #     correlation_value_temporal = self.softmax(tp)  # n, 16, 16
        #     tp = torch.einsum('nck,nkt->ntc', cube_feature.mean(3), correlation_value_temporal) # n 16(a) 256
        #     tp = dec(tp).permute(0,2,1)  # n 256 16(a)
        # # tp = tp + cube_feature.mean(3)

        # # concat
        # instance_feature = torch.concat((sp.mean(2), tp.mean(2)), dim=1)
        # instance_feature = self.relu(self.fc_spatial_temporal(instance_feature))  # project  n 256
        # instance_norm = F.normalize(instance_feature, p=2, dim=1)
        # tp = cube_feature.mean(3)
        # sp = self.relu(self.spatial_project(sp.mean(2))) # n 768
        # tp = self.relu(self.temporal_project(tp.mean(2))) # n 768
        # sp_norm = F.normalize(sp, p=2, dim=1)
        # tp_norm = F.normalize(tp, p=2, dim=1)
        # instance_feature = torch.concat((sp.unsqueeze(1), tp.unsqueeze(1)), dim=1)  # n 2 768
        # instance_norm = F.normalize(instance_feature, p=2, dim=2)
        # instance_feature = torch.concat((sp_proj_list[2], tp_proj_list[2]), dim=1) 
        # instance_feature = self.relu(self.fc_spatial_temporal(instance_feature))
        # instance_norm = F.normalize(instance_feature, p=2, dim=1)  # n 768
        # label_semantics = F.normalize(label_semantics, p=2, dim=-1)  # 55 768
        # logits_label = torch.einsum('nc,kc->nk',instance_norm,label_semantics)*0.1


        # semantic 
        spatial_fg_norm = F.normalize(spatial_round[2], p=2, dim=-1)   # 55(5) 10 768
        spatial_mg_norm = F.normalize(spatial_round[1], p=2, dim=-1)   # 55(5) 10 768
        spatial_cg_norm = F.normalize(spatial_round[0], p=2, dim=-1)   # 55(5) 10 768
        # spatial_label_norm = F.normalize(spatial_round[0], p=2, dim=-1)   # 55(5) 768
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
        logits_spatial_list.append(logits_spatial_cg*0.2)  # n 55  ntu60-0.1
        logits_spatial_list.append(logits_spatial_mg*0.2)  # n 55
        logits_spatial_list.append(logits_spatial_fg*0.2)  # n 55

        logits_temporal_list = []
        logits_temporal_cg = torch.einsum('nd,ckd->nck', tp_proj_list[0], temporal_cg_norm).topk(10, dim=2)[0].mean(2)
        logits_temporal_mg = torch.einsum('nd,ckd->nck', tp_proj_list[1], temporal_mg_norm).topk(10, dim=2)[0].mean(2) # n 55 10
        logits_temporal_fg = torch.einsum('nd,ckd->nck', tp_proj_list[2], temporal_fg_norm).topk(10, dim=2)[0].mean(2)
        logits_temporal_list.append(logits_temporal_cg*0.2)  # n 55
        logits_temporal_list.append(logits_temporal_mg*0.2)  # n 55
        logits_temporal_list.append(logits_temporal_fg*0.2)  # n 55
        # logits_temporal = F.softmax(logits_temporal_cg, dim=1) + F.softmax(logits_temporal_mg, dim=1) +F.softmax(logits_temporal_fg, dim=1)
        # logits_temporal = logits_temporal.mean(2) 

        return logits_spatial_list, logits_temporal_list, sp_proj_list, tp_proj_list, cube_feature, tp_attention_list

        # contrastive class prototype
        # cl_logits_list = []
        # rl_logits_list = []
        # if mode == "train":
        #     for idx, sp_feature in enumerate(sp_list):
        #         pred_value, pred_idx = torch.max(logits_list[idx], dim=1)  # n 1
        #         pred_oh = F.one_hot(pred_idx, num_classes=55)   # n 55
        #         label_oh = F.one_hot(label_idx, num_classes=55)
                
        #         tp = pred_oh * label_oh  # n 55
        #         tp_feature = torch.einsum('cn,nk->ck',sp_feature.mean(2).permute(1, 0), tp.float())  # 256 55
        #         tp_sum = tp.sum(dim=0, keepdim=True)  # 1 55
        #         tp_feature = tp_feature / (tp_sum + 1e-12)
        #         avg_f = self.avg_proptype[idx].detach().to(tp.device)  #   256 55
                
        #         has_object = (tp_sum > 1e-8).float()
        #         has_object[has_object > 0.1] = 0.9
        #         has_object[has_object <= 0.1] = 1.0
        #         f_mem = avg_f * has_object + (1 - has_object) * tp_feature
        #         f_mem = f_mem.to(tp.device)
        #         with torch.no_grad():
        #             self.avg_proptype[idx] = f_mem

        #         cl_logits = torch.einsum('nc,ck->nk',sp_feature.mean(2), f_mem)  # n 55
        #         cl_logits_list.append(cl_logits)
        #         if idx == 0:
        #             rl_logits = torch.einsum('kc,cm->km', self.proj_spatials[idx](f_mem.permute(1,0)), semantic_norm_list[idx].permute(1,0))
        #         else:
        #             rl_logits = torch.einsum('kc,cmn->kmn', self.proj_spatials[idx](f_mem.permute(1,0)), semantic_norm_list[idx].permute(2,0,1)).mean(2)
        #         rl_logits_list.append(rl_logits)
        #     return logits_list, cl_logits_list, rl_logits_list
        # else:
        #     return logits_list, sp_proj_list
    


        



