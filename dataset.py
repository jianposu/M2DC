import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle, randrange, randint
from torch import tensor, float32, save, load
from torch.utils.data import Dataset
import torch
from nilearn.image import load_img, smooth_img, clean_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal, fetch_atlas_destrieux_2009, fetch_atlas_harvard_oxford
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import h5py
import scipy.io as scio

class DatasetRestMDD(Dataset):
    def __init__(self, sourcedir, argv, roi, k_fold=None, target_feature='Gender', smoothing_fwhm=None):
        super().__init__()
        file_list, id_list, label_list = file_arrange(sourcedir, 'RestMDD')
        self.timeseries_dict = {}
        self.label_list = {}
        self.site_list = {}
        self.roi = roi
        sourcedir = Path(sourcedir)
        for i, file in enumerate(tqdm(file_list, ncols=60)):
            id = file.split('.')[0]
            filename = sourcedir / 'RestMDD_timecourse_1720' / file
            site = int(file.split('-')[0].split('S')[1])
            try:
                with h5py.File(filename, 'r',libver='latest', swmr=True) as temp:
                    timecourse = np.transpose(temp['timecourse'])
                    #label = temp['label']
            except:
                timecourse = scio.loadmat(filename)['timecourse']
            label = label_list[i]
            timeseries = (timecourse - np.mean(timecourse, axis=0, keepdims=True)) / np.std(timecourse, axis=0, keepdims=True)
            thres = len(np.where(np.isnan(timeseries))[0])
            if thres == 0:
                if self.roi == 'aal':
                    self.timeseries_dict[id] = np.nan_to_num(timeseries[:140,:116])
                elif self.roi == 'cc200':
                    self.timeseries_dict[id] = np.nan_to_num(timeseries[:140,116:316])
                elif self.roi == '980':
                    self.timeseries_dict[id] = np.nan_to_num(timeseries[:140,316:1296])
                self.label_list[id] = label
                self.site_list[id] = site
            else:
                print(id)

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())

        self.num_classes = 2
        self.full_label_list = label_list
        self.full_site_list = list(self.site_list.values())
        if k_fold is None:
            self.subject_list = self.full_subject_list
            
        else:
            testlist = argv.targetsite
            valilist = argv.validatesite
            trainlist = argv.sourcesite
            self.k_fold = {}
            for nfold in range(0,len(testlist)):
                self.k_fold[nfold] = []
                trainindex = [i for i in range(len(self.full_site_list)) if self.full_site_list[i] in trainlist]
                valiindex = [i for i in range(len(self.full_site_list)) if self.full_site_list[i]==valilist[nfold]]
                testindex = [i for i in range(len(self.full_site_list)) if self.full_site_list[i]==testlist[nfold]]
                self.k_fold[nfold].append(trainindex)
                self.k_fold[nfold].append(testindex)
                self.k_fold[nfold].append(valiindex) 
        self.k = None
        

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)


    def set_fold(self, fold, mode):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx, vali_idx = self.k_fold[fold]
        if mode == 'train': 
            self.subject_list = [self.full_subject_list[idx] for idx in train_idx] 
        if mode == 'test':
            self.subject_list = [self.full_subject_list[idx] for idx in test_idx]
        if mode == 'validate':
            self.subject_list = [self.full_subject_list[idx] for idx in vali_idx]


    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timecourse = np.nan_to_num(timeseries)
        label = self.label_list[subject]
        site = self.site_list[subject]
        if label==1:
            label = tensor(1)
        elif label==0:
            label = tensor(0)
        else:
            raise
        return {'id': subject, 'timeseries': tensor(timecourse, dtype=float32), 'label': label, 'site':site}


class metaRestMDD(Dataset):
    """
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, sourcedir, roi, num_src, num_tgt, sourcesite, targetsite, batchsz, n_way, k_shot, k_query, resize, masf=False, startidx=0):
        """

        :param root: root path of data
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.sourcedir = sourcedir
        self.num_src = num_src
        self.num_tgt = num_tgt
        self.targetsite = targetsite
        self.sourcesite = sourcesite
        self.randomcut = randomcut(0.5)
        self.randomnoise = randomnoise(0.5)
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        self.masf = masf
        self.roi = roi
        if self.roi == 'aal':
            self.num_nodes = 116
        elif self.roi == 'cc200':
            self.num_nodes = 200
        self.num_classes = 2
        print('shuffle b:%d, %d-way, %d-shot, %d-query' % (batchsz, n_way, k_shot, k_query))

        self.loaddata()  # load all data
        self.k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        self.split_idx = {}
        for i, (k, v) in enumerate(self.timeseries_dict.items()):
            if k in self.sourcesite:
                if self.masf:
                    self.split_idx[k] = []
                    self.split_idx[k].append(list(range(len(v))))
                else:
                    train_idx, test_idx = list(self.k_fold.split(v, self.label_list[k]))[0]
                    self.split_idx[k] = []
                    self.split_idx[k].append(train_idx)
                    self.split_idx[k].append(test_idx)
        self.cls_num = 2 # len(self.data)


    def loaddata(self):
        file_list, id_list, label_list = file_arrange(self.sourcedir, 'RestMDD')
        self.timeseries_dict = {}
        self.label_list = {}
        for file in tqdm(file_list, ncols=60):
            # id = file.split('.')[0]
            filename = Path(self.sourcedir) / 'RestMDD_timecourse_1720' / file
            try:
                with h5py.File(filename, 'r',libver='latest', swmr=True) as temp:
                    timecourse = np.transpose(temp['timecourse'])
                    label = temp['label']
            except:
                timecourse = scio.loadmat(filename)['timecourse']
                label = scio.loadmat(filename)['label']
            site = int(file.split('-')[0].split('S')[1])

            if label==1:
                label = 0
            elif label==-1:
                label = 1
            else:
                raise

            timecourse = (timecourse - np.mean(timecourse, axis=0, keepdims=True)) / (np.std(timecourse, axis=0, keepdims=True))
            if len(np.where(np.isnan(timecourse[:140,:316]))[0]) == 0:
                if self.roi == 'aal':
                    timeseries = timecourse[:140,:116]
                elif self.roi == 'cc200':
                    timeseries = timecourse[:140,116:316]
                elif self.roi == '980':
                    timeseries = timecourse[:140,316:1296]
                
                if site in self.timeseries_dict.keys():
                    self.timeseries_dict[site].append(np.nan_to_num(timeseries))
                    self.label_list[site].append(label)
                else:
                    self.timeseries_dict[site] = [np.nan_to_num(timeseries)]
                    self.label_list[site] = [label]

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0][0].shape
        self.k = None

    def create_batch(self, batchsz_train, batchsz_test):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.batchsz_train = batchsz_train
        self.batchsz_test = batchsz_test
        self.support_x_batch_train = []  # support set batch
        self.support_y_batch_train = []
        self.support_site_batch_train = []
        self.query_x_batch_train = []  # query set batch#
        self.query_y_batch_train = []
        self.query_site_batch_train = []

        selected_cls = list(range(self.cls_num))
        for b in tqdm(range(batchsz_train),ncols=60):  # for each batch
            selected_site = np.random.choice(self.sourcesite, self.num_src+self.num_tgt, False)  
            # no duplicate, 2 sourcesite for support and 1 for evaluation each time
            support_x = []
            support_y = []
            support_site = []
            query_x = []
            query_y = []
            query_site = []
            selected_support_site = selected_site[:self.num_src]
            selected_query_site = selected_site[self.num_src:]
            for site in selected_support_site:
                idx = self.split_idx[site][0]
                for cls in selected_cls:
                    tempindex = [i for i in idx if self.label_list[site][i]==cls]
                    selected_idx = np.random.choice(len(tempindex), self.k_shot, False)
                    np.random.shuffle(selected_idx)
                    indexDtrain = np.array(tempindex)[np.array(selected_idx)]  # idx for Dtrain
                    support_x.append(np.array(self.timeseries_dict[site])[indexDtrain].tolist())  
                    support_y.append(np.array(self.label_list[site])[indexDtrain].tolist())
                    support_site.append(site*np.ones(self.k_shot))

            for site in selected_query_site:
                idx = self.split_idx[site][0]
                for cls in selected_cls:
                    tempindex = [i for i in idx if self.label_list[site][i]==cls]
                    selected_idx = np.random.choice(len(tempindex), self.k_query, False)
                    np.random.shuffle(selected_idx)
                    indexDtest = np.array(tempindex)[np.array(selected_idx)]  # idx for Dtest
                    query_x.append(np.array(self.timeseries_dict[site])[indexDtest].tolist())
                    query_y.append(np.array(self.label_list[site])[indexDtest].tolist())
                    query_site.append(site*np.ones(self.k_query))

            self.support_x_batch_train.append(support_x)  # append set to current sets
            self.support_y_batch_train.append(support_y)
            self.support_site_batch_train.append(support_site)
            self.query_x_batch_train.append(query_x)  # append sets to current sets
            self.query_y_batch_train.append(query_y)
            self.query_site_batch_train.append(query_site)
        
    def set_mode(self, mode):
        if mode == 'train':
            self.mode = mode
            self.support_x_batch = self.support_x_batch_train
            self.support_y_batch = self.support_y_batch_train
            self.support_site_batch = self.support_site_batch_train
            self.query_x_batch = self.query_x_batch_train
            self.query_y_batch = self.query_y_batch_train
            self.query_site_batch = self.query_site_batch_train
            self.batchsz = self.batchsz_train  # batch of set, not batch of imgs
            self.setsz = self.num_src*self.n_way * self.k_shot  # num of samples per set
            self.querysz = self.num_tgt*self.n_way * self.k_query  # number of samples per set for evaluation
        else:
            self.mode = mode
            self.support_x_batch = self.support_x_batch_test
            self.support_y_batch = self.support_y_batch_test
            self.support_site_batch = self.support_y_batch_test #不需要用到，随便赋值
            self.query_x_batch = self.query_x_batch_test
            self.query_y_batch = self.query_y_batch_test
            self.query_site_batch = self.query_y_batch_test
            self.batchsz = self.batchsz_test  # batch of set, not batch of imgs
            self.setsz = self.num_src*self.n_way * self.k_shot  # num of samples per set
            self.querysz = self.num_tgt*self.n_way * self.k_query  # number of samples per set for evaluation

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, self.num_timepoints, self.num_nodes)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, self.num_timepoints, self.num_nodes)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        support_site = np.zeros((self.setsz,1), dtype=np.int)
        query_site = np.zeros((self.querysz,1), dtype=np.int)
        choice = 0 
        zero = 0
        i = 0
        for clslist in self.support_site_batch[index]:
            for item in clslist:
                support_site[i] = item
                i+=1
            
        i = 0
        for clslist in self.query_site_batch[index]:
            for item in clslist:
                query_site[i] = item
                i+=1

        i = 0
        for clslist in self.support_x_batch[index]:
            for item in clslist:
                #support_x[i] = torch.FloatTensor(item)
                
                if self.mode == 'test' or choice == 1:
                    support_x[i] = torch.FloatTensor(item)
                elif self.mode == 'train' and choice == 0:
                    support_x[i] = torch.FloatTensor(item)
                i+=1
        
        i = 0
        for clslist in self.support_y_batch[index]:
            for item in clslist:
                support_y[i] = item
                i+=1
        
        i = 0
        for clslist in self.query_x_batch[index]:
            for item in clslist:
                if self.mode == 'test' or choice == 0:
                    query_x[i] = torch.FloatTensor(item)
                elif self.mode == 'train' and choice == 1:
                    if zero == 1:
                        query_x[i] = self.randomcut(torch.FloatTensor(item))
                    else:
                        query_x[i] = self.randomnoise(torch.FloatTensor(item))
                i+=1
        
        i = 0
        for clslist in self.query_y_batch[index]:
            for item in clslist:
                query_y[i] = item
                i+=1

        return support_x, torch.LongTensor(support_y), support_site, query_x, torch.LongTensor(query_y), query_site


    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


def file_arrange(rootpath, dataset):
    rootpath = Path(rootpath)
    files = []
    labels = []
    id = []
    ffile = open(rootpath / (dataset+'_file_timecourse.txt'))
    flabel = open(rootpath / (dataset+'_label_timecourse.txt'))
    fid = open(rootpath / (dataset+'_id_timecourse.txt'))
    line_file = ffile.readline()
    line_label = flabel.readline()
    line_id = fid.readline()

    while line_file:
        tempfile = line_file.strip('\n')
        tempid = line_id.strip('\n')
        #tempid = tempid.split('-')[0]
        templabel = int(line_label.strip('\n'))
        id.append(tempid)
        files.append(tempfile)
        labels.append(templabel)
        line_file = ffile.readline()
        line_label = flabel.readline()
        line_id = fid.readline()

    ffile.close()
    flabel.close()
    fid.close()
    return files, id, np.array(labels).flatten()


class randomcut(object):
    def __init__(self,ratio):
        self.ratio = ratio
    
    def __call__(self,tensors):
        h, w = tensors.shape
        thres = int(w*self.ratio)
        numcut = randint(1,thres)
        ind = np.random.choice(list(range(w)), numcut, False)
        mat = torch.ones(h,w) 
        mat[:,ind] = 0
        return torch.mul(tensors,mat)


class randomnoise(object):
    def __init__(self,ratio):
        self.ratio = ratio
    
    def __call__(self,tensors):
        h, w = tensors.shape
        thres = int(w*(1-self.ratio))
        numcut = randint(thres,w)
        ind = np.random.choice(list(range(w)), numcut, False)
        mat = torch.randn(h,w) 
        mat[:,ind] = 0
        return tensors+mat