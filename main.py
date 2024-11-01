import util
from experiment import train, test
from m3l_experiment import meta_train
from analysis import analyze
import random
import numpy as np
import torch
import os

if __name__=='__main__':
    # parse options and make directories
    argv = util.option.parse()
    
    random.seed(argv.seed)
    np.random.seed(argv.seed)
    torch.manual_seed(argv.seed)
    torch.cuda.manual_seed(argv.seed)
    torch.cuda.manual_seed_all(argv.seed)
    # Use meta-learning or not
    if argv.meta:
        meta_train(argv)
    else:
        if not argv.no_train: train(argv)
        if not argv.no_test: test(argv)
        if not argv.no_analysis and argv.roi=='schaefer': analyze(argv)
    exit(0)
