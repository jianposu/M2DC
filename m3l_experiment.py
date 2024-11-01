import os
import util
import random
import torch
import numpy as np
from model import *
#from m3l_cluster_meta import *
#from m3l_meta_module import *
from m3l_module import *
from dataset import *
from tqdm import tqdm
import logging
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import csv
   

def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, attention, latent, reg_ortho = model(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)
    loss = criterion(logit, label.to(device))
    reg_ortho *= reg_lambda
    loss += reg_ortho

    # optimize model
    if optimizer is not None:
       optimizer.zero_grad()
       loss.backward()
       if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
       optimizer.step()
       if scheduler is not None:
           scheduler.step()

    return logit, loss, attention, latent


def setup_logging(opt):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = Path(opt.targetdir) / 'result_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def meta_train(argv):
    # make directories
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'summary'), exist_ok=True)
    setup_logging(argv) 
    logger = logging.getLogger()  
    # set seed and device
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    if torch.cuda.is_available():
        device = torch.device(argv.device)
        torch.cuda.manual_seed_all(argv.seed)
    else:
        device = torch.device("cpu")

    # define dataset
    metadataset = metaRestMDD(sourcedir=argv.sourcedir, roi=argv.roi, num_src = argv.num_src, num_tgt = argv.num_tgt,
                              sourcesite = argv.sourcesite, targetsite=argv.targetsite, n_way=argv.n_way, k_shot=argv.k_spt,
                        k_query=argv.k_qry, batchsz=10000, resize=argv.imgsz, masf=True)
    
    metadataset.create_batch(argv.train_batch,argv.test_batch)
    metadataset.set_mode('train')
    metadataloader = torch.utils.data.DataLoader(
        metadataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    
    dataset = DatasetRestMDD(argv.sourcedir, argv, roi=argv.roi, k_fold=argv.k_fold, smoothing_fwhm=argv.fwhm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, drop_last=False, shuffle=False, num_workers=0, pin_memory=True)

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(argv.targetdir, 'checkpoint.pth'), map_location=argv.device)
    else:
        checkpoint = {
            'epoch': 0,
            'model': None,}

    # start experiment
    # make directories per fold
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    # define model
    # meta-learning without distance constraint
    if argv.meta_method == 'm3l':
        maml = M3L(argv,metadataset.num_nodes,metadataset.cls_num,device)
    # meta-learning with distance constraint
    elif argv.meta_method == 'm3l_cluster':
        maml = M3L_cluster(argv,metadataset.num_nodes,metadataset.cls_num,device)
    maml.cuda()
    if checkpoint['model'] is not None: maml.net.load_state_dict(checkpoint['model'])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    best_loss_src = 10000.0
    best_loss_tgt = 10000.0
    best_test_acc = 0.0
    maml.update_lr = maml.lr_scheduler.get_lr()[0] 
    loss_q_list = []
    loss_kl_list = []
    loss_cluster_list = []
    loss_total_list = []
    acc_trn_list = []
    acc_tst_list = []
    auc_tst_list = []

    # start training
    for epoch in range(checkpoint['epoch'], argv.num_epochs):
        train_iter = len(metadataloader)
        argv.temp_para = argv.dis_para*min((1000**(float(epoch+10)/float(argv.num_epochs)))/1000,1.0)

        total_loss_src = []
        total_loss_tgt = []
        total_loss_cluster = []
        total_acc_tgt = []

        for i, (x_spt, y_spt, site_spt, x_qry, y_qry, site_qry) in enumerate(metadataloader):
            # process input data
            iters_start_time = time.time()
            network_bns = [x for x in list(maml.net.modules()) if isinstance(x, MixUpBatchNorm1d)]

            for bn in network_bns:
                bn.meta_mean1 = torch.zeros(bn.meta_mean1.size()).float().to(device)
                bn.meta_var1 = torch.zeros(bn.meta_var1.size()).float().to(device)
                bn.meta_mean2 = torch.zeros(bn.meta_mean2.size()).float().to(device)
                bn.meta_var2 = torch.zeros(bn.meta_var2.size()).float().to(device)
            
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            accs, loss_src, loss_tgt, loss_cluster = maml(x_spt, y_spt, site_spt, x_qry, y_qry, site_qry, criterion, metadataset.setsz, metadataset.querysz, argv,epoch)
            total_loss_src.append(loss_src)
            total_loss_tgt.append(loss_tgt)
            total_loss_cluster.append(loss_cluster)
            total_acc_tgt.append(accs)

            train_time = iters_start_time = time.time()-iters_start_time

            if i % 10 == 0:
                lr = maml.lr_scheduler.get_lr()[0]
                log_train = 'Epoch: {}/{}  Step: {}/{}  Time:{:.3f}s  LR: {:.9f}, Training Acc: {:.5f}, kl_loss: {:.5f}, Loss_q: {:.5f}, Loss_cluster: {:.5f}'.format(
                    epoch+1,argv.num_epochs,i,train_iter,train_time, lr, accs, loss_src, loss_tgt, loss_cluster)
                logger.info(log_train)
        maml.lr_scheduler.step()
        maml.update_lr = maml.lr_scheduler.get_lr()[0] 
        maml.tau *= maml.gamma_tau
        total_loss_src = np.stack(total_loss_src)
        total_loss_tgt = np.stack(total_loss_tgt)
        total_acc_tgt = np.stack(total_acc_tgt)
        total_loss_cluster = np.stack(total_loss_cluster)
        total_loss_src = total_loss_src.mean()
        total_loss_tgt = total_loss_tgt.mean()
        total_loss_cluster = total_loss_cluster.mean()
        total_acc_tgt = total_acc_tgt.mean()
        if total_loss_src < best_loss_src:
            best_loss_src = total_loss_src
        if total_loss_tgt < best_loss_tgt:
            best_loss_tgt = total_loss_tgt
        if total_acc_tgt > best_test_acc:
            best_test_acc = total_acc_tgt
        total = total_loss_src+total_loss_tgt+total_loss_cluster
        log_test = 'Epoch: {}/{}  kl_loss: {:.5f}, Best kl_loss: {:.5f}, Loss_q: {:.5f} , Best Loss_q: {:.5f} , Loss_cluster: {:.5f}, total_Loss:{:.5f}, ACC_tgt: {:.5f}, Best ACC_tgt: {:.5f}'.format(
            epoch+1,argv.num_epochs, total_loss_src, best_loss_src, total_loss_tgt, best_loss_tgt, total_loss_cluster, total, total_acc_tgt, best_test_acc)
        logger.info(log_test)

        loss_q_list.append(total_loss_tgt)
        loss_kl_list.append(total_loss_src)
        loss_cluster_list.append(total_loss_cluster)
        acc_trn_list.append(total_acc_tgt)
        loss_total_list.append(total)

        iters_start_time = time.time()
        dataset.set_fold(0,'test')
        fold_attention = {'node_attention': [], 'time_attention': []}
        loss_accumulate = 0.0
        latent_accumulate = []
        labellist = []
        predlist = []
        problist = []

        maml.net.eval()
        for i, x in enumerate(dataloader):
            with torch.no_grad():
                # process input data
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride,argv.dynamic_length)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(1,0,2)
                label = x['label']

                logit, loss, attention, latent = step(
                    model=maml.net,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None)
                pred = logit.argmax(1)
                labellist.append(label.detach().cpu().numpy())
                predlist.append(pred.detach().cpu().numpy())
                prob = logit.softmax(1)
                problist.append(prob.detach().cpu().numpy())
                loss_accumulate += loss.detach().cpu().numpy()

                fold_attention['node_attention'].append(attention['node-attention'].detach().cpu().numpy())
                fold_attention['time_attention'].append(attention['time-attention'].detach().cpu().numpy())
                latent_accumulate.append(latent.detach().cpu().numpy())
        labellist = np.concatenate(labellist)
        predlist = np.concatenate(predlist)
        problist = np.concatenate(problist)
        fpr,tpr,_= roc_curve(labellist,problist[:,1])
        tn, fp, fn, tp = confusion_matrix(labellist, predlist).ravel()
        filename = os.path.join(argv.targetdir,'training_result.csv')
        with open(filename, 'w+', newline='') as ff:
            csv_write = csv.writer(ff)         
            header = ['label', 'prob1', 'prob2']
            csv_write.writerow(header)
            for i in range(len(labellist)):
                data_row = [labellist[i], problist[i,0], problist[i,1]]
                csv_write.writerow(data_row)
        # 计算特异性和敏感性
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        roc_auc = auc(fpr,tpr)
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        # [b, update_step+1]
        test_time = time.time()-iters_start_time
        accs = float(np.sum(labellist==predlist))/len(labellist)
        log_test = 'Epoch: {}/{}  Time:{:.3f}s  Testing Acc: {:.5f}  Testing AUC: {:.5f}  Specificity: {:.5f}  Sensitivity: {:.5f} F1: {:.5f}'.format(
            epoch+1,argv.num_epochs,test_time, accs, roc_auc, specificity, sensitivity, f1)
        logger.info(log_test)
        acc_tst_list.append(accs)
        auc_tst_list.append(roc_auc)
        
        maml.net.train()
        
        # dataset.set_mode('train')
        # save checkpoint
        torch.save({
            'epoch': epoch+1,
            'model': maml.net.state_dict()},
            os.path.join(argv.targetdir, 'checkpoint.pth'))

    # finalize fold
    loss_q_list = np.vstack(loss_q_list)
    loss_kl_list = np.vstack(loss_kl_list)
    loss_cluster_list = np.vstack(loss_cluster_list)
    loss_total_list = np.vstack(loss_total_list)
    acc_trn_list = np.vstack(acc_trn_list)
    acc_tst_list = np.vstack(acc_tst_list)
    auc_tst_list = np.vstack(auc_tst_list)
    scale = np.linspace(1,argv.num_epochs, num=argv.num_epochs)
    result = {'loss_q': loss_q_list, 'loss_kl': loss_kl_list, 'loss_cluster':loss_cluster_list, 
                  'total_loss': loss_total_list, 'trn_acc': acc_trn_list, 'tst_acc': acc_tst_list,
                  'tst_auc': auc_tst_list}
    PlotFigure(argv, result, scale)
    filename1 = os.path.join(argv.targetdir,'training_info.csv')
    with open(filename1, 'a+', newline='') as ff:
        csv_write = csv.writer(ff)         
        header = ['epoch','Loss q','loss kl', 'loss cluster', 'loss total', 'Trn acc', 'Tst acc', 'Tst auc']
        csv_write.writerow(header)
        for i in range(argv.num_epochs):
            data_row = [i+1, loss_q_list[i][0], loss_kl_list[i][0], loss_cluster_list[i][0], 
                        loss_total_list[i][0], acc_trn_list[i][0], acc_tst_list[i][0], auc_tst_list[i][0]]
            csv_write.writerow(data_row) 
    
    torch.save(maml.net.state_dict(), os.path.join(argv.targetdir, 'model', 'model.pth'))
    checkpoint.update({'epoch': 0, 'model': None})

    os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))
