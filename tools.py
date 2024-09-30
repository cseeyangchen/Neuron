import numpy as np
import numpy
import torch.nn.functional as F
import torch
import torch.nn as nn

def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


def cross_modal_adaptive_loss(x1, x2, logit_scale, text_logits, rgb_logits, device):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # KL similarity
    kl_mean_x1_x2 = F.kl_div(F.log_softmax(rgb_logits, 1), F.softmax(text_logits, 1), reduction='mean')
    kl_mean_x2_x1 = F.kl_div(F.log_softmax(text_logits, 1), F.softmax(rgb_logits, 1), reduction='mean')
    kl_off_x1_x2 = torch.exp(-1*kl_mean_x1_x2)
    kl_off_x2_x1 = torch.exp(-1*kl_mean_x2_x1)

    # log_softmax
    n, _ = logits_per_x1.size()
    logits_x1_x2_final = -(torch.mul(kl_off_x1_x2, torch.diag(F.log_softmax(logits_per_x1, dim=1), 0))).cuda(device)
    logits_x2_x1_final = -(torch.mul(kl_off_x2_x1, torch.diag(F.log_softmax(logits_per_x2, dim=1), 0))).cuda(device)

    # result
    loss = sum(logits_x1_x2_final+logits_x2_x1_final)/(2*n)
    loss = loss.cuda(device)

    return loss





def cross_modal_loss(x1, x2, logit_scale, device):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # # KL similarity
    # kl_mean_x1_x2 = F.kl_div(F.log_softmax(rgb_logits, 1), F.softmax(text_logits, 1), reduction='mean')
    # kl_mean_x2_x1 = F.kl_div(F.log_softmax(text_logits, 1), F.softmax(rgb_logits, 1), reduction='mean')
    # kl_off_x1_x2 = torch.exp(-1*kl_mean_x1_x2)
    # kl_off_x2_x1 = torch.exp(-1*kl_mean_x2_x1)

    # log_softmax
    n, _ = logits_per_x1.size()
    logits_x1_x2_final = -torch.diag(F.log_softmax(logits_per_x1, dim=1), 0).cuda(device)
    logits_x2_x1_final = -torch.diag(F.log_softmax(logits_per_x2, dim=1), 0).cuda(device)

    # result
    loss = sum(logits_x1_x2_final+logits_x2_x1_final)/(2*n)
    loss = loss.cuda(device)

    return loss






def info_nce_loss(x1, x2, logit_scale, device):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_x1_x1 = logit_scale * x1 @ x1.t()
    logits_x2_x2 = logit_scale * x2 @ x2.t()
    logits_x1_x2 = logit_scale * x1 @ x2.t()
    logits_x2_x1 = logit_scale * x2 @ x1.t()

    # mask
    n, _ = logits_x1_x2.size()
    mask_x1_x1 = torch.ones((n, n))-torch.eye(n, n)
    mask_x1_x1 = mask_x1_x1.cuda(device)
    mask_x1_x2 = torch.ones((n, n))
    mask_x1_x2 = mask_x1_x2.cuda(device)
    mask_x1_x2_all = torch.cat((mask_x1_x2, mask_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_all = torch.cat((logits_x1_x2, logits_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_mask = logits_x1_x2_all * mask_x1_x2_all
    softmax_x1_x2 = F.softmax(logits_x1_x2_mask, 1).cuda(device)  # n 2n

    mask_x2_x2 = torch.ones((n, n))-torch.eye(n, n)
    mask_x2_x2 = mask_x2_x2.cuda(device)
    mask_x2_x1 = torch.ones((n, n))
    mask_x2_x1 = mask_x2_x1.cuda(device)
    mask_x2_x1_all = torch.cat((mask_x2_x1, mask_x2_x2), dim=1).cuda(device)  # n 2n
    logits_x2_x1_all = torch.cat((logits_x2_x1, logits_x2_x2), dim=1).cuda(device)  # n 2n
    logits_x2_x1_mask = logits_x2_x1_all * mask_x2_x1_all
    softmax_x2_x1 = F.softmax(logits_x2_x1_mask, 1).cuda(device)  # n 2n

    # selection
    logits_x1_x2_final = -torch.log(torch.diag(softmax_x1_x2[:,:n], 0)).cuda(device)
    logits_x2_x1_final = -torch.log(torch.diag(softmax_x2_x1[:,:n], 0)).cuda(device)

    # result
    loss = sum(logits_x1_x2_final+logits_x2_x1_final)/(2*n)
    loss = loss.cuda(device)

    return loss



def info_nce_adaptive_loss(x1, x2, logit_scale, text_logits, device):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_x1_x1 = logit_scale * x1 @ x1.t()
    logits_x2_x2 = logit_scale * x2 @ x2.t()
    logits_x1_x2 = logit_scale * x1 @ x2.t()
    logits_x2_x1 = logit_scale * x2 @ x1.t()

    # mask
    n, _ = logits_x1_x2.size()
    mask_x1_x1 = torch.ones((n, n))-torch.eye(n, n)
    mask_x1_x1 = mask_x1_x1.cuda(device)
    mask_x1_x2 = torch.ones((n, n))
    mask_x1_x2 = mask_x1_x2.cuda(device)
    mask_x1_x2_all = torch.cat((mask_x1_x2, mask_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_all = torch.cat((logits_x1_x2, logits_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_mask = logits_x1_x2_all * mask_x1_x2_all
    softmax_x1_x2 = F.softmax(logits_x1_x2_mask, 1).cuda(device)  # n 2n

    mask_x2_x2 = torch.ones((n, n))-torch.eye(n, n)
    mask_x2_x2 = mask_x2_x2.cuda(device)
    mask_x2_x1 = torch.ones((n, n))
    mask_x2_x1 = mask_x2_x1.cuda(device)
    mask_x2_x1_all = torch.cat((mask_x2_x1, mask_x2_x2), dim=1).cuda(device)  # n 2n
    logits_x2_x1_all = torch.cat((logits_x2_x1, logits_x2_x2), dim=1).cuda(device)  # n 2n
    logits_x2_x1_mask = logits_x2_x1_all * mask_x2_x1_all
    softmax_x2_x1 = F.softmax(logits_x2_x1_mask, 1).cuda(device)  # n 2n

    # selection
    text_logits_softmax = F.softmax(text_logits, 1).cuda(device)
    text_logits_off = torch.max(text_logits_softmax, 1)[0] - torch.min(text_logits_softmax, 1)[0]
    logits_x1_x2_final = -torch.log(torch.mul(torch.diag(softmax_x1_x2[:,:n], 0), text_logits_off)).cuda(device)
    logits_x2_x1_final = -torch.log(torch.mul(torch.diag(softmax_x2_x1[:,:n], 0), text_logits_off)).cuda(device)

    # result
    loss = sum(logits_x1_x2_final+logits_x2_x1_final)/(2*n)
    loss = loss.cuda(device)

    return loss



def info_nce_adaptive_threshold_loss(x1, x2, logit_scale, text_logits, device):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_x1_x1 = logit_scale * x1 @ x1.t()
    logits_x2_x2 = logit_scale * x2 @ x2.t()
    logits_x1_x2 = logit_scale * x1 @ x2.t()
    logits_x2_x1 = logit_scale * x2 @ x1.t()

    # mask
    n, _ = logits_x1_x2.size()
    mask_x1_x1 = torch.ones((n, n))-torch.eye(n, n)
    mask_x1_x1 = mask_x1_x1.cuda(device)
    mask_x1_x2 = torch.ones((n, n))
    mask_x1_x2 = mask_x1_x2.cuda(device)
    mask_x1_x2_all = torch.cat((mask_x1_x2, mask_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_all = torch.cat((logits_x1_x2, logits_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_mask = logits_x1_x2_all * mask_x1_x2_all
    softmax_x1_x2 = F.softmax(logits_x1_x2_mask, 1).cuda(device)  # n 2n

    mask_x2_x2 = torch.ones((n, n))-torch.eye(n, n)
    mask_x2_x2 = mask_x2_x2.cuda(device)
    mask_x2_x1 = torch.ones((n, n))
    mask_x2_x1 = mask_x2_x1.cuda(device)
    mask_x2_x1_all = torch.cat((mask_x2_x1, mask_x2_x2), dim=1).cuda(device)  # n 2n
    logits_x2_x1_all = torch.cat((logits_x2_x1, logits_x2_x2), dim=1).cuda(device)  # n 2n
    logits_x2_x1_mask = logits_x2_x1_all * mask_x2_x1_all
    softmax_x2_x1 = F.softmax(logits_x2_x1_mask, 1).cuda(device)  # n 2n

    # selection
    text_logits_softmax = F.softmax(text_logits, 1).cuda(device)
    text_logits_top2 = torch.topk(text_logits_softmax, 2, 1)[0]
    text_logits_top2_diff = text_logits_top2[:, 0] - text_logits_top2[:, 1]
    text_logits_off = torch.max(text_logits_softmax, 1)[0]
    # text_logits_template = text_logits_softmax
    # text_logits_max = torch.max(text_logits_template, 1)[0]
    # text_logits_template[text_logits_template==text_logits_max] = 0
    # text_logits_second_max = torch.max(text_logits_template, 1)[0]
    # text_logits_off = text_logits_max - text_logits_second_max
    threshold = 0.1
    text_logits_off[text_logits_top2_diff < threshold] = 0.0001
    logits_x1_x2_final = -torch.log(torch.mul(torch.diag(softmax_x1_x2[:,:n], 0), text_logits_off)).cuda(device)
    logits_x2_x1_final = -torch.log(torch.mul(torch.diag(softmax_x2_x1[:,:n], 0), text_logits_off)).cuda(device)

    # result
    loss = sum(logits_x1_x2_final+logits_x2_x1_final)/(2*n)
    loss = loss.cuda(device)

    return loss



def info_nce_adaptive_x2_loss(x1, x2, logit_scale, text_logits, device):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_x1_x1 = logit_scale * x1 @ x1.t()
    logits_x2_x2 = logit_scale * x2 @ x2.t()
    logits_x1_x2 = logit_scale * x1 @ x2.t()
    logits_x2_x1 = logit_scale * x2 @ x1.t()

    # mask
    n, _ = logits_x1_x2.size()
    mask_x1_x1 = torch.ones((n, n))-torch.eye(n, n)
    mask_x1_x1 = mask_x1_x1.cuda(device)
    mask_x1_x2 = torch.ones((n, n))
    mask_x1_x2 = mask_x1_x2.cuda(device)
    mask_x1_x2_all = torch.cat((mask_x1_x2, mask_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_all = torch.cat((logits_x1_x2, logits_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_mask = logits_x1_x2_all * mask_x1_x2_all
    softmax_x1_x2 = F.softmax(logits_x1_x2_mask, 1).cuda(device)  # n 2n

    mask_x2_x2 = torch.ones((n, n))-torch.eye(n, n)
    mask_x2_x2 = mask_x2_x2.cuda(device)
    mask_x2_x1 = torch.ones((n, n))
    mask_x2_x1 = mask_x2_x1.cuda(device)
    mask_x2_x1_all = torch.cat((mask_x2_x1, mask_x2_x2), dim=1).cuda(device)  # n 2n
    logits_x2_x1_all = torch.cat((logits_x2_x1, logits_x2_x2), dim=1).cuda(device)  # n 2n
    logits_x2_x1_mask = logits_x2_x1_all * mask_x2_x1_all
    softmax_x2_x1 = F.softmax(logits_x2_x1_mask, 1).cuda(device)  # n 2n

    # selection
    text_logits_softmax = F.softmax(text_logits, 1).cuda(device)
    text_logits_off = torch.pow((torch.max(text_logits_softmax, 1)[0] - torch.min(text_logits_softmax, 1)[0]), 2)
    logits_x1_x2_final = -torch.log(torch.mul(torch.diag(softmax_x1_x2[:,:n], 0), text_logits_off)).cuda(device)
    logits_x2_x1_final = -torch.log(torch.mul(torch.diag(softmax_x2_x1[:,:n], 0), text_logits_off)).cuda(device)

    # result
    loss = sum(logits_x1_x2_final+logits_x2_x1_final)/(2*n)
    loss = loss.cuda(device)

    return loss


def topk_loss(x1, x2, logit_scale, k, device):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)   # n, c
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity
    logits_x1_x2 = logit_scale * x1 @ x2.t()   # n, n
    logits_x2_x1 = logit_scale * x2 @ x1.t()
    logits_x1_x1 = logit_scale * x1 @ x1.t()
    logits_x2_x2 = logit_scale * x2 @ x2.t()

    # calculate topk
    n, _ = logits_x1_x2.size()
    topk_mask = torch.ones((n, n))-torch.eye(n, n)
    topk_mask = topk_mask.cuda(device)  
    topk_index = torch.argsort(logits_x2_x2 * topk_mask, dim=1, descending=True).cuda(device)   

    # mask selection
    mask_x1_x1 = torch.ones((n, n))-torch.eye(n, n)
    mask_x1_x1 = mask_x1_x1.cuda(device)
    mask_x1_x2 = torch.ones((n, n))
    mask_x1_x2 = mask_x1_x2.cuda(device)
    mask_x1_x2_all = torch.cat((mask_x1_x2, mask_x1_x1), dim=1).cuda(device)    # n 2n
    logits_x1_x2_all = torch.cat((logits_x1_x2, logits_x1_x1), dim=1).cuda(device)  # n 2n
    logits_x1_x2_mask = logits_x1_x2_all * mask_x1_x2_all
    log_softmax_x1_x2 = F.softmax(logits_x1_x2_mask, 1).cuda(device)  # n 2n

    mask_x2_x2 = torch.ones((n, n))-torch.eye(n, n)
    mask_x2_x2 = mask_x2_x2.cuda(device)
    mask_x2_x1 = torch.ones((n, n))
    mask_x2_x1 = mask_x2_x1.cuda(device)
    mask_x2_x1_all = torch.cat((mask_x2_x2, mask_x2_x1), dim=1).cuda(device)  # n 2n
    logits_x2_x1_all = torch.cat((logits_x2_x2, logits_x2_x1), dim=1) .cuda(device) # n 2n
    logits_x2_x1_mask = logits_x2_x1_all * mask_x2_x1_all
    log_softmax_x2_x1 = F.softmax(logits_x2_x1_mask, 1).cuda(device)  # n 2n

    # topk selection
    rows = torch.cat([i* torch.ones((1, k+1)) for i in range(n)]).type(torch.int).cuda(device)
    cols = torch.cat((torch.arange(0, n).type(torch.int).view(n, 1).cuda(device), topk_index[:,0:k]), dim=1).cuda(device)
    logits_x1_x2_topk = log_softmax_x1_x2[rows, cols]  # n k+1
    logits_x2_x1_topk = log_softmax_x2_x1[rows, cols]  # n k+1

    # result
    logits_x1_x2_final = -torch.log(torch.mean(logits_x1_x2_topk, dim=1)).cuda(device)
    logits_x2_x1_final = -torch.log(torch.mean(logits_x2_x1_topk, dim=1)).cuda(device)
    loss = sum(logits_x1_x2_final+logits_x2_x1_final)/(2*n)
    loss = loss.cuda(device)

    return loss




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand((200, 512)).cuda(device)
    y = torch.rand((200, 512)).cuda(device)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda(device)
    loss = topk_loss(x, y, logit_scale, 3, device)
    loss1 = info_nce_loss(x, y, logit_scale, device)
    loss3 = cross_modal_loss(x, y, logit_scale, device)
    print(loss)
    print(loss1)
    print(loss3)



