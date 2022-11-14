import csv
import os
import random
import time
import warnings

import dgl
import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from dataset import MetaDataset
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger('log')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False


def collate_fn_spt(samples):
    all_samples_list = list(map(list, zip(*samples)))
    triplets, labels = all_samples_list
    return triplets, labels


def collate_fn_train(samples):
    all_samples_list = list(map(list, zip(*samples)))
    input_graphs, tgt_graphs, support_edges, query_edges, tgt_events = all_samples_list
    return input_graphs, tgt_graphs, support_edges, query_edges, tgt_events


def collate_fn_test(samples):
    all_samples_list = list(map(list, zip(*samples)))
    input_graphs, tgt_graphs, support_edges, valid_edges, query_edges, tgt_events = all_samples_list
    return input_graphs, tgt_graphs, support_edges, valid_edges, query_edges, tgt_events


def load_dataset(args):
    t0 = time.time()
    path = os.path.join(args.data_dir, args.dataset, 'fold_{}'.format(str(args.fold)))

    idx2event = dict()
    with open(os.path.join(path, 'dict.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx2event[int(row['idx'])] = row['event']
    event2idx = {event: idx for idx, event in idx2event.items()}
    args.idx2event = idx2event
    args.event2idx = event2idx

    table = PrettyTable(['Index', 'Graph', '# Nodes', '# Edges', 'Avg Degree', '% Nodes', 'Type'])
    all_df = None
    max_num_nodes = 0
    split2event = dict()
    event2link = dict()

    for split in ['train', 'test']:
        split_path = os.path.join(path, '{}.csv'.format(split))
        split_df = pd.read_csv(split_path)
        split_src, split_dst = split_df['user_1'].tolist(), split_df['user_2'].tolist()
        split2event_idx = list(set(split_df['event'].tolist()))
        split2event[split] = [idx2event[x] for x in split2event_idx]

        max_num_nodes = max(max_num_nodes, max(max(split_src), max(split_dst)) + 1)
        all_df = pd.concat([all_df, split_df])

        for event in split2event[split]:
            event_idx = event2idx[event]
            event_df = split_df[split_df['event'] == event_idx]
            src = np.array(event_df['user_1'].tolist())
            dst = np.array(event_df['user_2'].tolist())

            mask = src < dst
            src, dst = src[mask], dst[mask]
            event2link[event] = (src, dst)
            assert len(src) * 2 == len(event_df)

    event2split = dict()
    for split, events in split2event.items():
        for event in events:
            event2split[event] = split

    for idx, (event, (src, dst)) in enumerate(event2link.items()):
        n_nodes = len(set(dst.tolist()) | set(src.tolist()))
        n_links = len(src)
        table.add_row(
            [idx, event, n_nodes, 2 * n_links, 2 * n_links / n_nodes, n_nodes / max_num_nodes, event2split[event]])

    max_n_link_distinct = len(all_df[['user_1', 'user_2']].drop_duplicates())
    max_n_link_all = len(all_df)
    table.add_row(
        ['All', 'global', max_num_nodes, max_n_link_distinct, max_n_link_distinct / max_num_nodes, 1.0, 'distinct'])
    table.add_row(['All', 'global', max_num_nodes, max_n_link_all, max_n_link_all / max_num_nodes, 1.0, 'all'])
    logger.info('\n'+str(table))

    src_list = []
    dst_list = []
    rel_list = []
    for event, (src, dst) in event2link.items():
        src_list += list(src)
        dst_list += list(dst)
        rel_list += [event2idx[event]] * len(src)

    src_list = torch.tensor(src_list, dtype=torch.long)
    dst_list = torch.tensor(dst_list, dtype=torch.long)
    rel_list = torch.tensor(rel_list, dtype=torch.long)

    g = dgl.graph((src_list, dst_list), num_nodes=max_num_nodes)
    g.edata['etype'] = rel_list

    args.num_nodes = g.number_of_nodes()
    args.num_rels = len(set(rel_list.tolist()))

    # meta-learning datasets and dataloaders
    train_dataset = MetaDataset(args=args, mode='train', g=g, split2event=split2event,
                                num_sample_tasks=args.num_batches * args.num_tasks)
    test_dataset = MetaDataset(args=args, mode='test', g=g, split2event=split2event,
                               num_sample_tasks=args.num_trials * len(split2event['test']))

    test_dataset.sample()

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.num_tasks, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=False, collate_fn=collate_fn_train)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False, collate_fn=collate_fn_test)

    logger.info('Data  | Load data in {:.5f} s'.format(time.time() - t0))

    return train_loader, test_loader


def trainable_params_stats(model):
    # initialize the variables and statistics
    total = 0
    stats_table = PrettyTable(['Index', 'Param ', 'Size', '# Params', 'Trainable'])
    for idx, (name, param) in enumerate(model.named_parameters(recurse=True)):
        if param.requires_grad:
            stats_table.add_row([idx, name, param.size(), np.prod(param.shape), 'True'])
            total += np.prod(param.shape)
        else:
            stats_table.add_row([idx, name, param.size(), np.prod(param.shape), 'False'])

    logger.info('\n'+str(stats_table))
    logger.info('Total Trainable Parameters: {}'.format(total))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose=False, delta=0, metrics='loss', save_path=None):
        """
        :param patience: int, How long to wait after last time validation loss improved.
        :param verbose: bool, If True, prints a message for each validation loss improvement.
        :param delta: float, Minimum change in the monitored quantity to qualify as an improvement.
        :param metrics: str, descent or ascent
        :param save_path:
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

        self.best_score = 0
        self.metrics = metrics

        if self.metrics in ['auc', 'ap','f1', 'acc']:
            self.ascend = True
        elif self.metrics in ['loss', 'tgt_loss']:
            self.ascend = False
        else:
            raise NotImplementedError

    def __call__(self, log, model):
        if self.metrics in ['auc', 'ap','f1','p','r','acc']:
            if isinstance(log[self.metrics], tuple):
                val_value = log[self.metrics][0]
            else:
                val_value = log[self.metrics]
        elif self.metrics in ['loss', 'tgt_loss']:
            val_value = log[self.metrics]
        else:
            raise NotImplementedError

        score = val_value if self.ascend else -val_value

        if self.best_score == 0:
            if self.save_path is not None and model is not None:
                self.save_checkpoint(val_value, model)
            self.best_score = score
        elif score <= self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.save_path is not None:
                self.save_checkpoint(val_value, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_value, model):
        """Saves model when validation loss decrease."""
        if model is not None:
            if self.verbose:
                previous_score = self.best_score if self.ascend else -1 * self.best_score
                logger.info('Valid | {}:({:.06f} --> {:.06f}). Saving model to {}'
                      .format(self.metrics, previous_score, val_value, self.save_path))
            torch.save(model.state_dict(), self.save_path)

