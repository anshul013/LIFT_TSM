import argparse
import os
import random
import numpy as np
import torch
import copy
import datetime
import time
from exp.exp_main import Exp_Main
from exp.exp_main_l import Exp_Main_LIFT
from exp.exp_lead import Exp_Lead
from data_provider import data_loader
from settings import data_settings

# Add timestamp logging
ds = time.strftime("%Y%m%d", time.localtime())
dh = time.strftime("%Y%m%d%H", time.localtime())
cur_sec = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

fix_seed = 3000
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.autograd.set_detect_anomaly(True)
np.random.seed(fix_seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--wo_test', action='store_true', default=False, help='only valid, not test')
parser.add_argument('--only_test', action='store_true', default=False)
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TSMixer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# Mixers
parser.add_argument('--num_blocks', type=int, default=3, help='number of mixer blocks to be used in TSMixer')
parser.add_argument('--hidden_size', type=int, default=32, help='first dense layer diminsions for mlp features block')
parser.add_argument('--single_layer_mixer', type=str2bool, nargs='?', default=False, help="if true a single layer mixers are used")

#Common between Mixers and Transformer-based models
parser.add_argument('--activation', type=str, choices={"gelu", "relu", "linear"}, default='gelu', help='activation')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--early_stopping', type=str2bool, nargs='?',
                        const=True, default=True,
                        help="whether to include early stopping or not")
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--norm', type=str, choices={"batch", "instance"}, default="batch", help="type of normalization")

# Patching and Convolution models
parser.add_argument('--exclude_inter_patch_mixing', type=str2bool, default=False, help='define if inter patch mixing is used for PatchTSMixer model')
parser.add_argument('--exclude_intra_patch_mixing', type=str2bool, default=False, help='define if intra patch mixing is used for PatchTSMixer model')
parser.add_argument('--exclude_channel_mixing', type=str2bool, default=False, help='define if channel mixing is used for PatchTSMixer model')
parser.add_argument('--patch_size', type=int, default=16, help="Number of timesteps per patch")
parser.add_argument('--kernel_size', type=int, default=1, help="conv width to cover certain number of timesteps")
parser.add_argument('--stride', type=int, default=1, help='number of non-overlapping timesteps for conv operation')
parser.add_argument('--affine', type=str2bool, default=True, help='define if the rev_norm is affine or not')
parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dimension for models including patch embedding")

# LIFT
parser.add_argument('--leader_num', type=int, default=4, help='# of leaders')
parser.add_argument('--state_num', type=int, default=8, help='# of variate states')
parser.add_argument('--prefetch_path', type=str, default='./prefetch/', help='location of prefetch files that records lead-lag relationships')
parser.add_argument('--tag', type=str, default='_max')
parser.add_argument('--prefetch_batch_size', type=int, default=16, help='prefetch_batch_size')
parser.add_argument('--variable_batch_size', type=int, default=32, help='variable_batch_size')
parser.add_argument('--max_leader_num', type=int, default=16, help='max # of leaders')
parser.add_argument('--masked_corr', action='store_true', default=False)
parser.add_argument('--efficient', type=str_to_bool, default=True)
parser.add_argument('--pin_gpu', type=str_to_bool, default=True)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
parser.add_argument('--lift', action='store_true', default=False)
parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
parser.add_argument('--border_type', type=str, default=None, help='border type for the model')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument("--use_gpu", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="use gpu")
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# SciNet
parser.add_argument("--num_levels", type=int, default=3)
parser.add_argument("--num_decoder_layer", type=int, default=1)
parser.add_argument("--concat_len", type=int, default=0)
parser.add_argument("--groups", type=int, default=3)
parser.add_argument("--kernel", type=int, default=3)
parser.add_argument("--single_step_output_One", type=int, default=3)
parser.add_argument("--positionalE",  type=str2bool, nargs='?', const=False, default=False)
parser.add_argument("--modified",  type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--RIN",  type=str2bool, nargs='?', const=False, default=False)


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# LIFT-specific data settings
if args.lift:
    args.enc_in, args.c_out = data_settings[args.data][args.features]
    args.data_path = data_settings[args.data]['data']
    args.dec_in = args.enc_in
    
    if args.tag and args.tag[0] != '_':
        args.tag = '_' + args.tag

    # Handle prefetch path
    K_tag = f'_K{args.leader_num}' if args.leader_num > 8 and args.enc_in > 8 else ''
    prefetch_path = os.path.join(args.prefetch_path, f'{args.data}_L{args.seq_len}{K_tag}{args.tag}')
    if not os.path.exists(prefetch_path + '_train.npz'):
        K_tag = f'_K16' if args.leader_num > 8 and args.enc_in > 8 else ''
        prefetch_path = os.path.join(args.prefetch_path, f'{args.data}_L{args.seq_len}{K_tag}{args.tag}')
    args.prefetch_path = prefetch_path

print('Args in experiment:')
print(args)

# LIFT experiment selection
FLAG_LIFT = args.lift
if FLAG_LIFT:
    Exp = Exp_Lead
    args.wrap_data_class = [data_loader.Dataset_Lead_Pretrain if args.freeze else data_loader.Dataset_Lead]
    if args.data.startswith('ETT'):
        args.efficient = False
else:
    Exp = Exp_Main_LIFT

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    train_data, train_loader, vali_data, vali_loader = None, None, None, None
    test_data, test_loader = None, None

    if args.is_training:
        all_results = {'mse': [], 'mae': []}
        for ii in range(args.itr):
            fix_seed = 3000 + ii
            setup_seed(fix_seed)
            print('Seed:', fix_seed)

            if args.model == 'TSMixer':
                setting = '{}_{}_{}_ft{}_sl{}_pl{}_hs{}_nb{}_dr{}_act{}_{}'.format(
                    args.model_id,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.hidden_size,
                    args.num_blocks,
                    args.dropout,
                    args.activation,
                    ii)
            else:
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.model_id,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    ii)

            if args.pretrain:
                if args.model == 'TSMixer':
                    pretrain_setting = '{}_{}_{}_ft{}_sl{}_pl{}_hs{}_nb{}_dr{}_act{}_{}'.format(
                        args.model_id,
                        args.border_type if args.border_type else args.data,
                        args.features,
                        args.seq_len,
                        args.label_len,
                        args.pred_len,
                        args.hidden_size,
                        args.num_blocks,
                        args.dropout,
                        args.activation,
                        ii)
                else:
                    pretrain_setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                        args.model_id,
                        args.border_type if args.border_type else args.data,
                        args.features,
                        args.seq_len,
                        args.label_len,
                        args.pred_len,
                        args.learning_rate,
                        args.d_model,
                        args.n_heads,
                        args.e_layers,
                        args.d_layers,
                        args.d_ff,
                        args.factor,
                        args.embed,
                        args.distil,
                        args.des, ii)
                args.pred_path = os.path.join('./results/', pretrain_setting, 'real_prediction.npy')
                args.load_path = os.path.join('./checkpoints/', pretrain_setting, 'checkpoint.pth')
                if FLAG_LIFT and args.freeze:
                    if not os.path.exists(args.pred_path):
                        _args = copy.deepcopy(args)
                        _args.freeze = False
                        _args.wrap_data_class = []
                        exp = Exp_Main(_args)
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(pretrain_setting))
                        exp.predict(pretrain_setting, True)
                        torch.cuda.empty_cache()

            if args.lift:
                setting += '_lift'

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            _, train_data, train_loader, vali_data, vali_loader = exp.train(setting, train_data, train_loader, vali_data, vali_loader)
            torch.cuda.empty_cache()

            if not args.wo_test and not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                mse, mae, test_data, test_loader = exp.test(setting, test_data, test_loader)
                all_results['mse'].append(mse)
                all_results['mae'].append(mae)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

        if not args.wo_test and not args.train_only:
            for k in all_results.keys():
                all_results[k] = np.array(all_results[k])
                all_results[k] = [all_results[k].mean(), all_results[k].std()]
            print('Final Results:')
            print(all_results)
    else:
        ii = 0
        if args.model == 'TSMixer':
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_hs{}_nb{}_dr{}_act{}_{}'.format(
                args.model_id,
                args.border_type if args.border_type else args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.hidden_size,
                args.num_blocks,
                args.dropout,
                args.activation,
                ii)
        else:
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.border_type if args.border_type else args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.learning_rate,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

        if args.lift:
            setting += '_lift'

        exp = Exp(args)  # set experiments
        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
