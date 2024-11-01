from ast import arg
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
from torch.nn import functional as F
from scipy.stats import wasserstein_distance


class cosine_distance(nn.Module):
    def forward(self, tensor1, tensor2):
        norm_tensor1 = tensor1/tensor1.norm(dim=1,keepdim=True)
        norm_tensor2 = tensor2/tensor2.norm(dim=0,keepdim=True)
        return torch.matmul(norm_tensor1,norm_tensor2)

class M3L_cluster(nn.Module):
    """
    Meta Learner
    """
    def __init__(self,args,num_nodes,cls_num,device):
        """

        :param args:
        """
        super(M3L_cluster, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.lr_inner = args.lr_inner
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.reg_lambda = args.reg_lambda
        self.device = device
        self.reg_dg = args.reg_dg
        self.tau = args.tau
        self.gamma_tau = args.gamma_tau
        #self.embed = RCB_block(args.num_layers*(args.hidden_dim+2*args.top_k)*3, args.embed_dim)
        self.embed = RCB_block(3*4*args.hidden_dim, args.embed_dim)
        #self.domain_classifier = domain_module(args.num_layers*(args.hidden_dim+2*args.top_k))
        self.num_src = args.num_src
        self.num_tgt = args.num_tgt
        warmup_epochs = args.warmup_epochs
        num_epochs = args.num_epochs

        self.net = metaBNSTModelSTAGIN(
            input_dim=num_nodes,
            hidden_dim=args.hidden_dim,
            num_classes=cls_num,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            sparsity=args.sparsity,
            dropout=args.dropout,
            metaBN=args.metaBN,
            cls_token=args.cls_token,
            readout=args.readout,
            top_k=args.top_k)

        self.meta_optim = optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.embed_optim = optim.Adam(self.embed.parameters(), lr=args.lr_phi)

        if args.scheduler == 'ExponentialLR':
            self.lr_scheduler = lr_scheduler.ExponentialLR(self.meta_optim, args.gamma)
            self.lr_scheduler_embed = lr_scheduler.ExponentialLR(self.embed_optim, args.gamma)
        elif args.scheduler == 'MultiStepLR':
            self.lr_scheduler = lr_scheduler.MultiStepLR(self.meta_optim, args.multistep)
            self.lr_scheduler_embed = lr_scheduler.MultiStepLR(self.embed_optim, args.multistep)


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
        if args.metaBN:
            y_qry = y_qry.repeat(args.num_src)

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

            logits_s, _, _, reg_ortho = self.net(dyn_v_spt.to(self.device), dyn_a_spt.to(self.device), 
                t_spt.to(self.device), sampling_endpoints_spt, MTE='', save_index=save_index)
            logits_s = logits_s-torch.max(logits_s, 1)[0][:, None]
            loss_s = criterion(logits_s/self.tau, y_spt[sindex[1]])
            losses_src += loss_s
            save_index += 1

        losses_src /= self.num_src
        self.net.zero_grad()
        with torch.autograd.detect_anomaly():
            grad = torch.autograd.grad(losses_src, self.net.parameters(), retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - ((self.update_lr/100)) * p[0], zip(grad, self.net.parameters())))  
        self.net.copy_param(fast_weights)
        
        latents = {}
        logits_spt = {}
        logits_qry = {}
        #latent_spt = []
        loss_recon = 0.0
        criterion_mse = nn.MSELoss(reduction='mean')
        for c in classes:
            latents[c] = []
            logits_spt[c] = []
            logits_qry[c] = []
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

            logits_s, _, latent_s, reg_ortho_s = self.net(dyn_v_spt.to(self.device), dyn_a_spt.to(self.device), 
                t_spt.to(self.device), sampling_endpoints_spt)
            
            latent_s_encode, latent_s_decode = self.embed(latent_s)
            loss_recon += criterion_mse(latent_s, latent_s_decode)

            for c in classes:
                index = np.where(y_spt[sindex[1]].detach().cpu().numpy()==c)
                latents[c].append(latent_s_encode[index[0]])
                temp = logits_s[index[0]]/args.tau
                logits_spt[c].append(temp.softmax(1).mean(dim=0))
        loss_recon /= self.num_src
        
        dyn_a_qry, sampling_points_qry = util.bold.process_dynamic_fc(
            x_qry, args.window_size, args.window_stride, args.dynamic_length)
        # 1. run the i-th task and compute loss for k=0
        sampling_endpoints_qry = [p+args.window_size for p in sampling_points_qry]
        dyn_v_qry = repeat(torch.eye(num_nodes), 'n1 n2 -> b t n1 n2', 
            t=len(sampling_points_qry), b=querysz)
        if len(dyn_a_qry) < querysz: dyn_v_qry = dyn_v_qry[:len(dyn_a_qry)]
        t_qry = x_qry.permute(1,0,2)
        
        if args.metaBN:
            logits_q, _, latent_q, reg_ortho_q = self.net(
                dyn_v_qry.to(self.device), dyn_a_qry.to(self.device), t_qry.to(self.device), sampling_endpoints_qry, MTE='sample')
        else:
            logits_q, _, latent_q, reg_ortho_q = self.net(
                dyn_v_qry.to(self.device), dyn_a_qry.to(self.device), t_qry.to(self.device), sampling_endpoints_qry)
        
        logits_q = logits_q-torch.max(logits_q, 1)[0][:, None]
        latent_q_encode, latent_q_decode = self.embed(latent_q)
        loss_recon += criterion_mse(latent_q, latent_q_decode)

        for c in classes:
            index = np.where(y_qry.detach().cpu().numpy()==c)
            latents[c].append(latent_q_encode[index[0]])
            latents[c] = torch.cat(latents[c], dim=0)
            for i in range(args.num_src):
                temp = logits_q[index[0][i*args.k_qry:(i+1)*args.k_qry]]
                temp /= args.tau
                logits_qry[c].append(temp.softmax(1).mean(dim=0))
            logits_qry[c] = torch.cat(logits_qry[c], dim=0)
            logits_spt[c] = torch.cat(logits_spt[c], dim=0)
            
        kl_loss = 0.0

        kl_loss *= args.kl_para

        ii = args.num_src*args.k_spt
        disfunc = cosine_distance()
        intra = []
        for c in classes:
            dismat = disfunc(latents[c],latents[c].t())
            tempmat = dismat[ii:,:ii]
            intra.append(tempmat)
        intra = torch.cat(intra,dim=0)
        inter = torch.cdist(latents[0],latents[1],p=2)
        inter = torch.ones_like(inter)*args.dis_thres-inter
        inter = torch.where(inter>0,inter,torch.zeros_like(inter))
        
        loss_q = criterion(logits_q/self.tau, y_qry)
        loss_cluster = args.dis_para*(intra.mean()+inter.pow(2).mean()+loss_recon)
        losses_tgt = loss_q/self.num_tgt+loss_cluster
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects += correct

        # optimize theta parameters
        self.net.zero_grad()
        self.meta_optim.zero_grad()
        with torch.autograd.detect_anomaly():
            losses_tgt.backward(retain_graph=True)
        nn.utils.clip_grad_norm(self.net.parameters(), args.clip_grad, norm_type=2)
        self.embed.zero_grad()
        self.embed_optim.zero_grad()
        loss_cluster.backward()
        nn.utils.clip_grad_norm(self.embed.parameters(), args.clip_grad, norm_type=2)
        self.meta_optim.step()
        self.embed_optim.step()
        if args.metaBN:
            accs = corrects / ((self.num_src)*querysz)
        else:
            accs = corrects / ((self.num_tgt)*querysz)

        return accs, kl_loss, loss_q.detach().cpu().numpy(), loss_cluster.detach().cpu().numpy()


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
        t_qry = x

        logits_q, _, _, _ = self.net(
            dyn_v_qry.to(self.device), dyn_a_qry.to(self.device), t_qry.to(self.device), sampling_endpoints_qry, list(self.net.parameters()))

        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()

        return correct


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
        self.lr_inner = args.lr_inner
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.device = device
        self.reg_dg = args.reg_dg
        self.tau = args.tau
        self.gamma_tau = args.gamma_tau
        self.num_src = args.num_src
        self.num_tgt = args.num_tgt
        warmup_epochs = args.warmup_epochs
        num_epochs = args.num_epochs

        self.net = metaBNSTModelSTAGIN(
            input_dim=num_nodes,
            hidden_dim=args.hidden_dim,
            num_classes=cls_num,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            sparsity=args.sparsity,
            dropout=args.dropout,
            metaBN=args.metaBN,
            cls_token=args.cls_token,
            readout=args.readout,
            top_k=args.top_k)

        self.meta_optim = optim.Adam(self.net.parameters(), lr=args.lr)
        self.lr_scheduler = lr_scheduler.ExponentialLR(self.meta_optim, args.gamma)


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
        if args.metaBN:
            y_qry = y_qry.repeat(args.num_src)

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

            logits_s, _, _, _ = self.net(dyn_v_spt.to(self.device), dyn_a_spt.to(self.device), 
                t_spt.to(self.device), sampling_endpoints_spt, MTE='', save_index=save_index)
            loss_s = criterion(logits_s/self.tau, y_spt[sindex[1]])
            losses_src += loss_s
            save_index += 1

        losses_src /= 2
        self.net.zero_grad()
        grad = torch.autograd.grad(losses_src, self.net.parameters(),retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - ((self.update_lr/100)) * p[0], zip(grad, self.net.parameters())))  
        self.net.copy_param(fast_weights)
       
        dyn_a_qry, sampling_points_qry = util.bold.process_dynamic_fc(
            x_qry, args.window_size, args.window_stride, args.dynamic_length)
        # 1. run the i-th task and compute loss for k=0
        sampling_endpoints_qry = [p+args.window_size for p in sampling_points_qry]
        dyn_v_qry = repeat(torch.eye(num_nodes), 'n1 n2 -> b t n1 n2', 
            t=len(sampling_points_qry), b=querysz)
        if len(dyn_a_qry) < querysz: dyn_v_qry = dyn_v_qry[:len(dyn_a_qry)]
        t_qry = x_qry.permute(1,0,2)
        if args.metaBN:
            logits_q, _, _, _ = self.net(
                dyn_v_qry.to(self.device), dyn_a_qry.to(self.device), t_qry.to(self.device), sampling_endpoints_qry,  MTE='sample')
        else:
            logits_q, _, _, _ = self.net(
                dyn_v_qry.to(self.device), dyn_a_qry.to(self.device), t_qry.to(self.device), sampling_endpoints_qry)
            
        loss_q = criterion(logits_q/self.tau, y_qry)
        losses_tgt = loss_q
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects += correct

        # optimize theta parameters
        self.meta_optim.zero_grad()
        losses_tgt.backward()
        nn.utils.clip_grad_norm(self.net.parameters(), args.clip_grad, norm_type=2)
        self.meta_optim.step()
        if args.metaBN:
            accs = corrects / ((args.num_src)*querysz)
        else:
            accs = corrects / (querysz)

        return accs, losses_src.detach().cpu().numpy(), losses_tgt.detach().cpu().numpy(), loss_q.detach().cpu().numpy()


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
        t_qry = x 

        logits_q, _, _ = self.net(
            dyn_v_qry.to(self.device), dyn_a_qry.to(self.device), t_qry.to(self.device), sampling_endpoints_qry, list(self.net.parameters()))

        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()

        return correct