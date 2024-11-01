import os
import csv
import argparse
from datetime import datetime


def parse():
    parser = argparse.ArgumentParser(description='SPATIO-TEMPORAL-ATTENTION-GRAPH-ISOMORPHISM-NETWORK')
    parser.add_argument('-s', '--seed', type=int, default=111)
    parser.add_argument('-n', '--exp_name', type=str, default='M3L_cluster-RestMDD_cc200_esta5_128_target-21_s10q5_0.0005_warmup10_80epoch-notime-metaBN-layer4-clip1.0-step-embed8-8-disthres10-dispara0.01-src-333')
    parser.add_argument('-k', '--k_fold', type=int, default=1)
    parser.add_argument('-b', '--minibatch_size', type=int, default=10)

    parser.add_argument('-ds', '--sourcedir', type=str, default='/HOME/scz0abb/run/jianpo/classification_data/MDD/')
    parser.add_argument('-dtest', '--testdir', type=str, default='/HOME/scz0abb/run/jianpo/classification_data/MDD/')
    parser.add_argument('-dt', '--targetdir', type=str, default='/HOME/scz0abb/run/jianpo/dynamic_GNN/result_20240728/')

    parser.add_argument('--dataset', type=str, default='RestMDD', choices=['rest', 'task','HCPrest512','RestMDD'])
    parser.add_argument('--roi', type=str, default='aal', choices=['scahefer', 'aal', 'cc200','destrieux', 'harvard_oxford','BA512'])
    parser.add_argument('--fwhm', type=float, default=None)

    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--window_stride', type=int, default=5)
    parser.add_argument('--dynamic_length', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--k_neighbor', type=int, default=10)
    parser.add_argument('--m',help='memorybank_update_rate', type=int, default=0.8)
    parser.add_argument('--dis_thres',help='distance threshold', type=float, default=10.0)
    parser.add_argument('--dis_thres_intra',help='intra distance threshold', type=float, default=1)
    parser.add_argument('--dis_para',help='distance regularization parameter', type=float, default=0.01)
    parser.add_argument('--temp_para',help='distance regularization parameter', type=float, default=0.001)
    parser.add_argument('--embed_dim',help='embedding dimension', type=int, default=8)

    parser.add_argument('--num_src', type=int, default=2)
    parser.add_argument('--num_tgt', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_phi', type=float, default=5e-5)
    parser.add_argument('--lr_inner', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--gamma_tau', type=float, default=1)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--final_div_factor', type=float, default=1000)
    parser.add_argument('--reg_lambda', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--clip_grad', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--multistep', type=int,nargs='+', help='multistep for lr_scheduler', default=[10,25,40,50])
    parser.add_argument('--num_steps', type=int, default=400)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--nometa_epochs', type=int, default=60)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--sparsity', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--readout', type=str, default='sero', choices=['garo', 'sero', 'mean'])
    parser.add_argument('--cls_token', type=str, default='sum', choices=['sum', 'mean', 'param'])

    parser.add_argument('--num_clusters', type=int, default=2)
    parser.add_argument('--subsample', type=int, default=50)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--no_analysis', action='store_true')
    parser.add_argument('--meta', action='store_false')
    parser.add_argument('--metaBN', action='store_false')
    parser.add_argument('--internal', action='store_false')
    parser.add_argument('--meta_method', type=str, default='m3l_cluster', choices=['m3l', 'm3l_cluster'])
    parser.add_argument('--scheduler', type=str, default='ExponentialLR', choices=['ExponentialLR', 'MultiStepLR'])
    parser.add_argument('--comp_method', type=str, default='DANN')
    parser.add_argument('--allloss', help='if source loss is used to update model', action='store_false')

    parser.add_argument('--sourcesite', type=list, help='sourcesite', default=[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25])
    parser.add_argument('--validatesite', type=int,nargs='+',  help='validatesite', default=[25])
    parser.add_argument('--targetsite', type=int, nargs='+', help='targetsite', default=[25])

    parser.add_argument('--testsite', type=str, nargs='+', help='testsite', default=['AMU','FMMU','FMMU-LI','SMU','Xiangya-Li','Xiangya-Pu'])
    parser.add_argument('--n_way', type=int, help='n way', default=2)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    parser.add_argument('--imgc', type=int, help='imgc', default=1)
    parser.add_argument('--train_batch', type=int, help='training batch', default=400)
    parser.add_argument('--test_batch', type=int, help='testing batch', default=40)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=5e-4)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
    parser.add_argument('--reg_dg', type=float, help='regularization parameter of domain generalization', default=0.1)
    parser.add_argument('--tau', type=float, help='temperature coefficient', default=0.5)
    parser.add_argument('--label_smooth', type=float, help='label smoothing', default=0)
    parser.add_argument('--kl_para', type=float, help='kl loss coefficient', default=0)

    argv = parser.parse_args()
    if argv.internal:
        argv.sourcesite = [i for i in argv.sourcesite if i is not argv.targetsite[0]]
        argv.validatesite = argv.targetsite
    
    if argv.meta:
        if argv.meta_method == 'm3l_cluster':
            argv.exp_name = datetime.now().strftime("%y-%m-%d-%H-%M")+'-M3L-cluster-cosine-distance-inner100'+'-exponential0.9-'+argv.roi+'-'+str(argv.hidden_dim)+'-target-'+str(argv.targetsite[0])+'-num_src-'+str(argv.num_src)+'-num_tgt-'+str(argv.num_tgt)+'-lr-'+str(argv.lr)+'-lr_phi-'+str(argv.lr_phi)+'-epoch-'+str(argv.num_epochs)+'-stride-'+str(argv.window_stride)+'-dis-'+str(argv.dis_thres)+'-'+str(argv.dis_para)+'-tau-'+str(argv.tau)+'-layer-'+str(argv.num_layers)
        else:
            argv.exp_name = datetime.now().strftime("%y-%m-%d-%H-%M")+'RDM-experiment-'+'-target-'+str(argv.testsite[0])+'-'+argv.roi+'-'+str(argv.hidden_dim)+'-num_src-'+str(argv.num_src)+'-epoch-'+str(argv.num_epochs)+'-stride-'+str(argv.window_stride)+'-tau-'+str(argv.tau)+'-layer-'+str(argv.num_layers)+'-test-'
        if not argv.internal:
            argv.exp_name = argv.exp_name+'-test'

        if argv.metaBN:
            argv.exp_name = argv.exp_name+'-metaBN-'+str(argv.seed)
        else:
            argv.exp_name = argv.exp_name+'-'+str(argv.seed)

    else:
        argv.exp_name = datetime.now().strftime("%y-%m-%d-%H-%M")+'-CSTAGIN1-'+argv.roi+'-'+str(argv.hidden_dim)+'-target-'+str(argv.targetsite[0])+'-lr-'+str(argv.lr)+'-epoch-'+str(argv.num_epochs)+'-stride-'+str(argv.window_stride)+'-tau-'+str(argv.tau)+'-gamma-'+str(argv.gamma_tau)+'-layer-'+str(argv.num_layers)+'-'+str(argv.seed)

    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir, 'argv.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv
