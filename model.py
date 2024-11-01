from turtle import forward
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import init
from torch.nn import functional as F
from torchvision.models import resnet50, resnet34
from collections import OrderedDict
from torch.autograd import Variable
import math
from torch.optim.lr_scheduler import LambdaLR
from MetaModules import MetaModule
from torch.autograd import Function
import torch.autograd as autograd
import copy
import operator
from numbers import Number
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def cal_clip_norm(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == np.inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = (p.data+1e-20).norm(norm_type)
            total_norm += param_norm.item()**norm_type
        total_norm = total_norm**(1./norm_type)
    clip_coef = max_norm/(total_norm+1e-6)
    return clip_coef

def PlotFigure(opt, result, scale):
    fig = plt.figure(1)
    font = {'family' : 'serif', 'color'  : 'black', 'weight' : 'bold', 'size'   : 16,}
    
    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(scale, result['loss_q'], 'r', label='Loss Q')
    ln2 = ax1.plot(scale,result['loss_kl'], 'k', label='KL Loss')
    ln3 = ax1.plot(scale,result['loss_cluster'], 'b', label='Loss Cluster')
    ln4 = ax1.plot(scale,result['total_loss'], 'g', label='Total Loss')
    ax2 = ax1.twinx()
    ln5 = ax2.plot(scale, result['trn_acc'], 'b--', label='Training ACC')
    ln6 = ax2.plot(scale, result['tst_acc'], 'k--', label='Testing ACC')
    ln7 = ax2.plot(scale, result['tst_auc'], 'g--', label='Testing AUC')

    lns = ln1+ ln2+ ln3+ ln4 + ln5+ ln6+ ln7
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=7)
    ax1.set_ylabel('Loss', fontdict=font)
    ax1.set_title("Training Curve", fontdict=font)
    ax1.set_xlabel('Epoch', fontdict=font)

    ax2.set_ylabel('Accuracy', fontdict=font)

    #figname = opt.targetdir / ('Training_curve.png')
    figname = os.path.join(opt.targetdir,'Training_curve.png')
    fig.savefig(figname)
    #print('Figure %s is saved.' % figname)
    plt.close(fig)
    
        
class ModuleTimestamping(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        #self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(10), nn.ReLU())

    def forward(self, t, sampling_endpoints):
        #return self.encoder(t[sampling_endpoints])
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p-1 for p in sampling_endpoints]]


class RNNencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        #self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(10), nn.ReLU())

    def forward(self, t):
        #return self.encoder(t[sampling_endpoints])
        return self.rnn(t)[0][:]
    

class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())


    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)


class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0, top_k=0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)
        self.top_k=top_k


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1]))
        x_graphattention = self.attend(x_embed)
        x_eff_graphattention = torch.sort(x_graphattention,dim=-1,descending=True)
        if self.top_k>0:
            x_eff_graphattention = x_eff_graphattention.values[:,:self.top_k].view(*x_shape[:-1],-1)
        x_graphattention = torch.sigmoid(x_graphattention).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2), x_eff_graphattention


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)


class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1, top_k=0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))
        self.top_k=top_k


    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_eff_attend = torch.sort(attn_matrix,dim=-1,descending=True)
        x_shape = x_attend.shape
        if self.top_k>0:
            x_eff_attend = x_eff_attend.values[:,:,:self.top_k].view(*x_shape[:-1],-1)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix, x_eff_attend


class simpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))


    def forward(self, x):
        x_attend, _ = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, X, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = X.size()[1]
        assert X.size()[1] <= max_seq_len
        l, d = X.size()[1], X.size()[-1]
        # P_{i,2j}   = sin(i/10000^{2j/d})
        # P_{i,2j+1} = cos(i/10000^{2j/d})
        # for i=0,1,...,l-1 and j=0,1,2,...,[(d-2)/2]
        max_seq_len = int((max_seq_len//l)*l)
        P = np.zeros([1, l, d])
        # T = i/10000^{2j/d}
        T = [i*1.0/10000**(2*j*1.0/d) for i in range(0, max_seq_len, max_seq_len//l) for j in range((d+1)//2)]
        T = np.array(T).reshape([l, (d+1)//2])
        if d % 2 != 0:
            P[0, :, 1::2] = np.cos(T[:, :-1])
        else:
            P[0, :, 1::2] = np.cos(T)
        P[0, :, 0::2] = np.sin(T)
        return torch.tensor(P, dtype=torch.float, device=X.device)


class transformer(nn.Module):
    def __init__(self, depth, timepoints, num_nodes, hidden_dim, num_heads=1, pool = 'mean', dropout=0.5, top_k=0, n_classes=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(simpleTransformer(input_dim = num_nodes, hidden_dim = hidden_dim, num_heads = num_heads))
        self.pool = pool
        self.pos_embedding = nn.Parameter(torch.randn(1, timepoints + 1, num_nodes, requires_grad=True)) # learnable positional embedding
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, num_nodes, requires_grad=True))
        self.dim = num_nodes
        self.timepoints = timepoints
        #self.to_latent = nn.Identity()
    
    def forward(self, x, sampling_endpoints):
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        y = x+self.pos_embedding[:, :(self.timepoints + 1)]
        x = self.dropout(y)
        for layer in self.layers:
            x = layer(x)
        
        return x[:,list(np.array(sampling_endpoints)+1)]


class mlp_head(nn.Module):
    def __init__(self, num_nodes, n_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(num_nodes),nn.Linear(num_nodes, n_classes))

    def forward(self, x):
        x = x.mean(dim = 1)
        return self.mlp(x)

class metaBNSTModelSTAGIN(nn.Module):
#class metaBNSTModelSTAGIN(MetaModule):    
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, metaBN=True, dropout=0.5, cls_token='sum', readout='sero', garo_upscale=1.0, top_k=0):
        super(metaBNSTModelSTAGIN,self).__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        if readout=='garo': readout_module = ModuleGARO
        elif readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        else: raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None
        self.num_classes = num_classes
        self.sparsity = sparsity
        self.num_layers = num_layers

        # define modules
        self.time_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim)
        self.initial_linear = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.metaBN = metaBN
        #if self.metaBN:
        self.feat_bn = nn.ModuleList() 

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1, top_k=top_k))
            self.transformer_modules.append(ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=num_heads, dropout=0.1, top_k=top_k))
            self.linear_layers.append(nn.Linear(3*hidden_dim, num_classes))
            mixupmodule = MixUpBatchNorm1d(3*hidden_dim)
            init.constant_(mixupmodule.weight, 1)
            init.constant_(mixupmodule.bias, 0)
            self.feat_bn.append(mixupmodule)
    
        
    def _collate_adjacency(self, a, sparse=True):
        i_list = []
        v_list = []
        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100-self.sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))
    
    
    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def copyModel(self, newModel, same_var=False):
        # copy meta model to meta model
        tarName = list(map(lambda v: v, newModel.state_dict().keys()))

        # requires_grad
        partName, partW = list(map(lambda v: v[0], newModel.named_params(newModel))), list(
            map(lambda v: v[1], newModel.named_params(newModel)))  # new model's weight

        metaName, metaW = list(map(lambda v: v[0], self.named_params(self))), list(
            map(lambda v: v[1], self.named_params(self)))
        bnNames = list(set(tarName) - set(partName))

        # copy vars
        for name, param in zip(metaName, partW):
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(self, name, param)
        # copy training mean var
        tarName = newModel.state_dict()
        for name in bnNames:
            param = to_var(tarName[name], requires_grad=False)
            self.setBN(self, name, param)
    
    
    def update_params(self, lr_inner, source_params=None,
                      solver='sgd', beta1=0.9, beta2=0.999, weight_decay=5e-4):
        if solver == 'sgd':
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src if src is not None else 0
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        elif solver == 'adam':
            for tgt, gradVal in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                exp_avg, exp_avg_sq = torch.zeros_like(param_t.data), \
                                      torch.zeros_like(param_t.data)
                bias_correction1 = 1 - beta1
                bias_correction2 = 1 - beta2
                gradVal.add_(weight_decay, param_t)
                exp_avg.mul_(beta1).add_(1 - beta1, gradVal)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, gradVal, gradVal)
                exp_avg_sq.add_(1e-8)  # to avoid possible nan in backward
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                step_size = lr_inner / bias_correction1
                newParam = param_t.addcdiv(-step_size, exp_avg, denom)
                self.set_param(self, name_t, newParam)


    def setParams(self, params):
        for tgt, param in zip(self.named_params(self), params):
            name_t, _ = tgt
            self.set_param(self, name_t, param)


    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)


    def setBN(self, inPart, name, param):
        if '.' in name:
            part = name.split('.')
            self.setBN(getattr(inPart, part[0]), '.'.join(part[1:]), param)
        else:
            setattr(inPart, name, param)


    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())
    
    
    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    
    def copy_param(self, weights):
        i = 0
        for param in self.parameters():
            param.data = weights[i]
            i+=1
    
    
    def forward(self, v, a, t, sampling_endpoints, MTE='', save_index=0):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a, actually it is [time x minibatch x node x node]
        logit = 0.0
        reg_ortho = 0.0
        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]

        time_encoding = torch.nan_to_num(self.time_encoder(t,sampling_endpoints))
        
        time_encoding = repeat(time_encoding, 't b c -> b t n c', n=num_nodes)

        h = torch.cat([v, time_encoding], dim=3)
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)

        a = self._collate_adjacency(a)
        if self.metaBN:
            for layer, (G, R, T, L, F) in enumerate(zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers, self.feat_bn)):
                h = G(h, a)
                h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
                h_readout, node_attn, node_attn_eff = R(h_bridge, node_axis=2)
                if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1,h_readout.shape[1],-1)])
                h_attend, time_attn, time_attn_eff = T(h_readout)
                ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')
                matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0,2,1))
                reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean()
                h_total = torch.cat((h_bridge.mean(dim=2),h_readout,h_attend),2)
                latent1 = self.cls_token(h_total)
                latent = F(latent1, MTE, save_index)
                
                if self.training:
                    if MTE is 'sample':
                        latent = torch.cat((latent[0],latent[1]),dim=0)
                        logit += self.dropout(L(latent))
                    else:
                        logit += self.dropout(L(latent))
                else:
                    logit += self.dropout(L(latent))
                
                latent_list.append(latent.clone())
                attention['node-attention'].append(node_attn)
                attention['time-attention'].append(time_attn)   

        else:
            for layer, (G, R, T, L, F) in enumerate(zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers, self.feat_bn)):
                h = G(h, a)
                h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
                h_readout, node_attn, node_attn_eff = R(h_bridge, node_axis=2)
                if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1,h_readout.shape[1],-1)])
                h_attend, time_attn, time_attn_eff = T(h_readout)
                ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')
                matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0,2,1))
                reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean()

                h_total = torch.cat((h_bridge.mean(dim=2),h_readout,h_attend),2)
                latent = self.cls_token(h_total)
                latent = F(latent, '', save_index)
                logit += self.dropout(L(latent))
                latent_list.append(latent.clone())
                attention['node-attention'].append(node_attn)
                attention['time-attention'].append(time_attn)           

        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        latent2 = torch.cat(latent_list, dim=1)
        return logit, attention, latent2, reg_ortho
    

def to_var(x, requires_grad=True):
    if torch.cuda.is_available(): x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, source_params=None,
                      solver='sgd', beta1=0.9, beta2=0.999, weight_decay=5e-4):
        if solver == 'sgd':
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src if src is not None else 0
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        elif solver == 'adam':
            for tgt, gradVal in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                exp_avg, exp_avg_sq = torch.zeros_like(param_t.data), \
                                      torch.zeros_like(param_t.data)
                bias_correction1 = 1 - beta1
                bias_correction2 = 1 - beta2
                gradVal.add_(weight_decay, param_t)
                exp_avg.mul_(beta1).add_(1 - beta1, gradVal)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, gradVal, gradVal)
                exp_avg_sq.add_(1e-8)  # to avoid possible nan in backward
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                step_size = lr_inner / bias_correction1
                newParam = param_t.addcdiv(-step_size, exp_avg, denom)
                self.set_param(self, name_t, newParam)

    def setParams(self, params):
        for tgt, param in zip(self.named_params(self), params):
            name_t, _ = tgt
            self.set_param(self, name_t, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def setBN(self, inPart, name, param):
        if '.' in name:
            part = name.split('.')
            self.setBN(getattr(inPart, part[0]), '.'.join(part[1:]), param)
        else:
            setattr(inPart, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copyModel(self, newModel, same_var=False):
        # copy meta model to meta model
        tarName = list(map(lambda v: v, newModel.state_dict().keys()))

        # requires_grad
        partName, partW = list(map(lambda v: v[0], newModel.named_params(newModel))), list(
            map(lambda v: v[1], newModel.named_params(newModel)))  # new model's weight

        metaName, metaW = list(map(lambda v: v[0], self.named_params(self))), list(
            map(lambda v: v[1], self.named_params(self)))
        bnNames = list(set(tarName) - set(partName))

        # copy vars
        for name, param in zip(metaName, partW):
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(self, name, param)
        # copy training mean var
        tarName = newModel.state_dict()
        for name in bnNames:
            param = to_var(tarName[name], requires_grad=False)
            self.setBN(self, name, param)

    def copyWeight(self, modelW):
        # copy state_dict to buffers
        curName = list(map(lambda v: v[0], self.named_params(self)))
        tarNames = set()
        for name in modelW.keys():
            # print(name)
            if name.startswith("module"):
                tarNames.add(".".join(name.split(".")[1:]))
            else:
                tarNames.add(name)
        bnNames = list(tarNames - set(curName))  
        for tgt in self.named_params(self):
            name_t, param_t = tgt
            # print(name_t)
            module_name_t = 'module.' + name_t
            if name_t in modelW:
                param = to_var(modelW[name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            elif module_name_t in modelW:
                param = to_var(modelW['module.' + name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            else:
                continue


    def copyWeight_eval(self, modelW):
        # copy state_dict to buffers
        curName = list(map(lambda v: v[0], self.named_params(self)))
        tarNames = set()
        for name in modelW.keys():
            # print(name)
            if name.startswith("module"):
                tarNames.add(".".join(name.split(".")[1:]))
            else:
                tarNames.add(name)
        bnNames = list(tarNames - set(curName))  ## in BN resMeta bnNames only contains running var/mean
        for tgt in self.named_params(self):
            name_t, param_t = tgt
            # print(name_t)
            module_name_t = 'module.' + name_t
            if name_t in modelW:
                param = to_var(modelW[name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            elif module_name_t in modelW:
                param = to_var(modelW['module.' + name_t], requires_grad=True)
                self.set_param(self, name_t, param)
            else:
                continue

        for name in bnNames:
            try:
                param = to_var(modelW[name], requires_grad=False)
            except:
                param = to_var(modelW['module.' + name], requires_grad=False)
            self.setBN(self, name, param)

    

class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.LongTensor([0]).squeeze())
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)
        ## meta test set this one to False self.training or not self.track_running_stats
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MixUpBatchNorm1d(MetaBatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MixUpBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('meta_mean1', torch.zeros(self.num_features))
        self.register_buffer('meta_var1', torch.zeros(self.num_features))
        self.register_buffer('meta_mean2', torch.zeros(self.num_features))
        self.register_buffer('meta_var2', torch.zeros(self.num_features))
        self.device_count = torch.cuda.device_count()

    def forward(self, input, MTE='', save_index=0):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            if MTE == 'sample':
                from torch.distributions.normal import Normal
                #print(self.meta_mean1,self.meta_var1,self.meta_mean2,self.meta_var2)
                Distri1 = Normal(self.meta_mean1, self.meta_var1)
                Distri2 = Normal(self.meta_mean2, self.meta_var2)
                sample1 = Distri1.sample([input.size(0), ])
                sample2 = Distri2.sample([input.size(0), ])
                lam = np.random.beta(1., 1.)
                inputmix1 = lam * sample1 + (1-lam) * input
                inputmix2 = lam * sample2 + (1-lam) * input

                mean1 = inputmix1.mean(dim=0)
                var1 = inputmix1.var(dim=0, unbiased=False)
                mean2 = inputmix2.mean(dim=0)
                var2 = inputmix2.var(dim=0, unbiased=False)

                output1 = (inputmix1 - mean1[None, :]) / (torch.sqrt(var1[None, :] + self.eps))
                output2 = (inputmix2 - mean2[None, :]) / (torch.sqrt(var2[None, :] + self.eps))
                if self.affine:
                    output1 = output1 * self.weight[None, :] + self.bias[None, :]
                    output2 = output2 * self.weight[None, :] + self.bias[None, :]
                return [output1,output2]

            else:
                mean = input.mean(dim=0)
                # use biased var in train
                var = input.var(dim=0, unbiased=False)
                n = input.numel() / input.size(1)

                with torch.no_grad():
                    running_mean = exponential_average_factor * mean \
                                   + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    running_var = exponential_average_factor * var * n / (n - 1) \
                                  + (1 - exponential_average_factor) * self.running_var
                    self.running_mean.copy_(running_mean)
                    self.running_var.copy_(running_var)
                    if save_index == 0:
                        self.meta_mean1.copy_(mean)
                        self.meta_var1.copy_(var)
                    elif save_index == 1:
                        self.meta_mean2.copy_(mean)
                        self.meta_var2.copy_(var)

        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input
   

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
