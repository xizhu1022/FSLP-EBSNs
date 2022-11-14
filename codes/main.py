import argparse
import time

import torch
import os

from meta import MetaLearning
from utils import seed_everything, load_dataset
import logging


def main():
    seed_everything(args.seed)

    train_loader, test_loader = load_dataset(args)

    meta_model = MetaLearning(args=args,
                              inner_lr=args.inner_lr,
                              outer_lr=args.outer_lr,
                              update_step=args.update_step,
                              update_step_test=args.update_step_test
                              )

    meta_model = meta_model.to(args.device)

    meta_model.run(train_loader, test_loader)

    logger.info(args)


def get_logger(args, verbosity=1, name='log'):
    # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

    _log_dir = '{}/{}/{}'.format(args.log_dir, args.dataset, args.fold)
    if not os.path.exists(_log_dir):
        os.makedirs(_log_dir, exist_ok=True)
    filename = '{}/{}_{}_{}_{}_{}_{}_{}.log'.format(_log_dir, args.t, args.encoder, args.decoder, args.binary,
                                                    args.train_ratio, args.aux_param, args.num_heads)
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    # "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, filename


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='DASFAA2023')

    argparser.add_argument('--dataset', type=str, default='dblp_1025', help='dataset')
    argparser.add_argument('--seed', type=int, default=114514)
    argparser.add_argument('--gpu', type=int, default=0)
    argparser.add_argument('--ckpt_dir', type=str, default='../checkpoint', help='Checkpoint dir')
    argparser.add_argument('--log_dir', type=str, default='../log', help='Checkpoint dir')
    argparser.add_argument('--data_dir', type=str, default='../data', help='Input data dir')
    argparser.add_argument('--fold', type=int, default=0, help='fold')

    # data
    argparser.add_argument('--num_batches', type=int, default=50, help='Number of sampling tasks in MetaDataset')
    argparser.add_argument('--num_tasks', type=int, default=4, help='Number of graphs/tasks')
    argparser.add_argument('--num_trials', type=int, default=1, help='Meta-testing times, only validation here')
    argparser.add_argument('--train_ratio', type=float, default=0.3, help='Support set from target graphs')
    argparser.add_argument('--valid_ratio', type=float, default=0.1, help='Validation set from target graphs')
    argparser.add_argument('--tgt_bsz', type=int, default=256, help='Batch size for target samples')
    argparser.add_argument('--aux_bsz', type=int, default=2048, help='Batch size for auxiliary samples')
    argparser.add_argument('--bsz', type=int, default=4096, help='Batch size for samples')
    argparser.add_argument('--type', type=str, default='split', help='Sample type (split/direct)')
    argparser.add_argument('--binary', type=bool, default=False, help='Edge types. Not used!')
    argparser.add_argument('--tgt_flag', type=bool, default=True, help='Use additional target event')
    argparser.add_argument('--aux_ratio', type=float, default=0.8, help='Network augmentation')  # (0.8,1.0)

    # model
    argparser.add_argument('--dim', type=int, default=32, help='Input feature dimension')
    argparser.add_argument('--num_layers', type=int, default=1, help='Number of GNN layers')
    argparser.add_argument('--num_heads', type=int, default=4, help='Number of heads')  # paper: 2
    argparser.add_argument('--aux_param', type=float, default=0.8, help='Auxiliary param')
    argparser.add_argument('--reg_param', type=float, default=0.01, help='Regularization param')
    argparser.add_argument('--encoder', type=str, default='hgb', help='Choose from hgb/gat')
    argparser.add_argument('--decoder', type=str, default='distmult', help='Choose from dot/distmult/rel')
    argparser.add_argument('--pooling', type=str, default='cat', help='Choose fro, cat/mean/last')

    # training
    argparser.add_argument('--max_steps', type=int, default=10001, help='Max step number')
    argparser.add_argument('--valid_freq', type=int, default=50, help='validation frequency')

    argparser.add_argument('--fast_adaption_flag',  default=False,action='store_true', help='Fast Adaption in meta-testing')
    argparser.add_argument('--update_step', type=int, default=10, help='Task-level local update steps')
    argparser.add_argument('--update_step_test', type=int, default=101, help='Steps for fine-tuning in meta-testing')
    argparser.add_argument('--max_fine_tuning_steps', type=int, default=100001, help='Max fine-tuning steps')
    argparser.add_argument('--inner_lr', type=float, default=1e-2, help='Task-level inner update learning rate')
    argparser.add_argument('--outer_lr', type=float, default=1e-2, help='Meta-level outer learning rate')
    argparser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    argparser.add_argument('--clip_weight_val', type=float, default=1, help='Clip weights std value')

    argparser.add_argument('--patience', type=int, default=3, help='Patience for model validation')
    argparser.add_argument('--metrics', type=str, default='auc', help='Validation metrics')
    argparser.add_argument('--use_d2', type=bool, default=False, help='Alibaba D2 platform')

    args = argparser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')

    if args.use_d2:
        args.data_dir = '/data/oss_bucket_0/jerry/data/out_data'
        args.ckpt_dir = '/data/oss_bucket_0/jerry/checkpoint'

    t = time.strftime("%Y%m%d%H%M%S", time.localtime())
    args.t = t
    args.save_postfix = '{}_{}_{}'.format(str(t), str(args.dataset), str(args.train_ratio))

    logger, filename = get_logger(args)
    logger.info(args)
    logger.info('Log File | {}'.format(filename))

    main()
