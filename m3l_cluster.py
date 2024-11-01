import os
import util
import random
import torch
import numpy as np
from model import *
from dataset import *
from tqdm import tqdm
from einops import repeat, rearrange
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import functional as F
from torch.optim import lr_scheduler
from copy import deepcopy
from torch import nn
import itertools
from scipy.spatial.distance import pdist,cdist

class M3L(nn.Module):
    """
    Meta Learner
    """
    def __init__(self,args,num_nodes,cls_num,device):
        """

        :param args:
        """
        super(M3L, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.device = device
        self.reg_dg = args.reg_dg
        self.embed = RCB_block(args.num_layers*(args.hidden_dim+2*args.top_k), args.embed_dim)
        warmup_epochs = args.warmup_epochs
        num_epochs = args.num_epochs

        self.net = metaBNModelSTAGIN(
            input_dim=num_nodes,
            hidden_dim=args.hidden_dim,
            num_classes=cls_num,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            sparsity=args.sparsity,
            dropout=args.dropout,
            cls_token=args.cls_token,
            readout=args.readout,
            top_k=args.top_k)
        '''
        self.metanet = metaBNModelSTAGIN(
            input_dim=num_nodes,
            hidden_dim=args.hidden_dim,
            num_classes=cls_num,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            sparsity=args.sparsity,
            dropout=args.dropout,
            cls_token=args.cls_token,
            readout=args.readout,
            top_k=args.top_k)
        '''
        #self.net.to(self.device)
        self.meta_optim = optim.Adam(itertools.chain(self.net.parameters(),self.embed.parameters()), lr=args.lr)
        #self.lr_scheduler = get_linear_schedule_with_warmup(self.meta_optim, warmup_epochs*args.num_steps, num_epochs*args.num_steps)
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.meta_optim, args.multistep)
        #self.lr_scheduler = lr_scheduler.OneCycleLR(self.meta_optim, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=args.num_steps, pct_start=args.pct_start, div_factor=args.max_lr/args.lr, final_div_factor=args.final_div_factor)
        #self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.meta_optim, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=args.num_steps, pct_start=0.2, div_factor=args.max_lr/args.lr, final_div_factor=1000)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def update_weight(self, net_param, params):
        i = 0
        for param in net_param:
            param.data = params[i]
            i+=1


    def forward(self, x_spt, y_spt, site_spt, x_qry, y_qry, site_qry, criterion, setsz, querysz, args, epoch):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        setsz, timepoints, num_nodes = x_spt.size()
        querysz = x_qry.size(0)
        n_site_src = len(np.unique(site_spt))
        site_src = np.unique(site_spt)
        classes = np.unique(y_spt.detach().cpu().numpy())
        #y_qry = y_qry.repeat(2)

        losses_src = 0
        losses_tgt = 0
        corrects = 0
        save_index = 0
        
        
        for s in site_src:
            sindex = np.where(site_spt == s)
            dyn_a_spt, sampling_points_spt = util.bold.process_dynamic_fc(
                x_spt[sindex[1]], args.window_size, args.window_stride, args.dynamic_length)
            # 1. run the i-th task and compute loss for k=0
            sampling_endpoints_spt = [p+args.window_size for p in sampling_points_spt]
            dyn_v_spt = repeat(torch.eye(num_nodes), 'n1 n2 -> b t n1 n2', 
                t=len(sampling_points_spt), b=setsz)
            if len(dyn_a_spt) < setsz: dyn_v_spt = dyn_v_spt[:len(dyn_a_spt)]
            t_spt = x_spt[sindex[1]].permute(1,0,2)

            logits_s, _, _ = self.net(dyn_v_spt.to(self.device), dyn_a_spt.to(self.device), 
                t_spt.to(self.device), sampling_endpoints_spt, MTE='', save_index=save_index)
            loss_s = criterion(logits_s, y_spt[sindex[1]])
            #reg_ortho_s1 = reg_ortho_s*args.reg_lambda
            losses_src += loss_s#(loss_s+reg_ortho_s1)
            save_index += 1

        losses_src /= 2
        self.net.zero_grad()
        grad = torch.autograd.grad(losses_src, self.net.parameters(),retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))  
        #self.metanet.update_params(lr_inner=self.update_lr, source_params=grad, solver='adam')
        self.net.copy_param(fast_weights)
        
        '''
        net_network_bns = [x for x in list(self.net.modules()) if isinstance(x, MixUpBatchNorm1d)]
        metanet_network_bns = [x for x in list(self.metanet.modules()) if isinstance(x, MixUpBatchNorm1d)]
        
        for i, bn in enumerate(net_network_bns):
            metanet_network_bns[i].meta_mean1 = bn.meta_mean1
            metanet_network_bns[i].meta_mean2 = bn.meta_mean2
            metanet_network_bns[i].meta_var1 = bn.meta_var1
            metanet_network_bns[i].meta_var2 = bn.meta_var2
        '''
        
        latents = {}
        for c in classes:
            latents[c] = []
        for s in site_src:
            sindex = np.where(site_spt == s)
            dyn_a_spt, sampling_points_spt = util.bold.process_dynamic_fc(
                x_spt[sindex[1]], args.window_size, args.window_stride, args.dynamic_length)
            # 1. run the i-th task and compute loss for k=0
            sampling_endpoints_spt = [p+args.window_size for p in sampling_points_spt]
            dyn_v_spt = repeat(torch.eye(num_nodes), 'n1 n2 -> b t n1 n2', 
                t=len(sampling_points_spt), b=setsz)
            if len(dyn_a_spt) < setsz: dyn_v_spt = dyn_v_spt[:len(dyn_a_spt)]
            t_spt = x_spt[sindex[1]].permute(1,0,2)

            logits_s, _, latent_s = self.net(dyn_v_spt.to(self.device), dyn_a_spt.to(self.device), 
                t_spt.to(self.device), sampling_endpoints_spt)
            
            latent_s = self.embed(latent_s)
            
            for c in classes:
                index = np.where(y_spt[sindex[1]].detach().cpu().numpy()==c)
                latents[c].append(latent_s[index[0]])
        
        dyn_a_qry, sampling_points_qry = util.bold.process_dynamic_fc(
            x_qry, args.window_size, args.window_stride, args.dynamic_length)
        # 1. run the i-th task and compute loss for k=0
        sampling_endpoints_qry = [p+args.window_size for p in sampling_points_qry]
        dyn_v_qry = repeat(torch.eye(num_nodes), 'n1 n2 -> b t n1 n2', 
            t=len(sampling_points_qry), b=querysz)
        if len(dyn_a_qry) < querysz: dyn_v_qry = dyn_v_qry[:len(dyn_a_qry)]
        t_qry = x_qry.permute(1,0,2)
        
        logits_q, _, latent_q = self.net(
            dyn_v_qry.to(self.device), dyn_a_qry.to(self.device), t_qry.to(self.device), sampling_endpoints_qry)#, MTE='sample')
                
        latent_q = self.embed(latent_q)
        
        for c in classes:
            index = np.where(y_qry.detach().cpu().numpy()==c)
            latents[c].append(latent_q[index[0]])
            latents[c] = torch.cat(latents[c], dim=0)
        
        intra = []
        for c in classes:
            dismat = torch.norm(latents[c][:,None]-latents[c],dim=2,p=2)
            tempmat = dismat[10:,:10]
            tempmat, index = torch.sort(tempmat, descending=False, dim=-1)
            intra.append(tempmat[:,:args.k_neighbor])
        intra = torch.cat(intra,dim=0)
        #intra = F.pdist(latents[0],p=2)+F.pdist(latents[1],p=2)
        inter = torch.cdist(latents[0],latents[1],p=2)
        inter = torch.ones_like(inter)*args.dis_thres-inter
        inter = torch.where(inter>0,inter,torch.zeros_like(inter))
        
        loss_q = criterion(logits_q, y_qry)
        loss_cluster = args.dis_para*(intra.mean()+inter.pow(2).mean())
        #reg_ortho_q1 = reg_ortho_q*args.reg_lambda
        losses_tgt = loss_q+loss_cluster
        #losses_tgt = loss_q+args.dis_para*(intra.mean()/inter.mean())
        #+losses_src#+reg_ortho_q1
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects += correct

        # optimize theta parameters
        self.meta_optim.zero_grad()
        losses_tgt.backward()
        nn.utils.clip_grad_norm(self.net.parameters(), args.clip_grad, norm_type=2)
        self.meta_optim.step()
        #self.lr_scheduler.step()
        accs = corrects / (querysz)

        return accs, losses_src.detach().cpu().numpy(), loss_q.detach().cpu().numpy(), loss_cluster.detach().cpu().numpy()


    def evaluate(self, x, y, args):

        corrects = 0
        querysz, timepoints, num_nodes = x.size()
        dyn_a_qry, sampling_points_qry = util.bold.process_dynamic_fc(
            x, args.window_size, args.window_stride, args.dynamic_length)
        # 1. run the i-th task and compute loss for k=0
        sampling_endpoints_qry = [p+args.window_size for p in sampling_points_qry]
        dyn_v_qry = repeat(torch.eye(num_nodes), 'n1 n2 -> b t n1 n2', 
            t=len(sampling_points_qry), b=querysz)
        if len(dyn_a_qry) < querysz: dyn_v_qry = dyn_v_qry[:len(dyn_a_qry)]
        t_qry = x #.permute(1,0,2)

        logits_q, _, _ = self.net(
            dyn_v_qry.to(self.device), dyn_a_qry.to(self.device), t_qry.to(self.device), sampling_endpoints_qry, list(self.net.parameters()))

        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()

        return correct