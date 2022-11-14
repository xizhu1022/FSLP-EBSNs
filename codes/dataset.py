from collections import defaultdict as ddict

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
import math

from dgl.nn.pytorch import GATConv

class MetaDataset(Dataset):
    def __init__(self, args, mode, g, split2event, num_sample_tasks):
        super(MetaDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.g = g
        self.split2event = split2event
        self.num_sample_tasks = num_sample_tasks
        self.event2tgt_sg, self.event2aux_sg = self.build_subgraphs()

    def build_subgraphs(self):
        event2tgt_sg = ddict()
        event2aux_sg = ddict()

        for event in self.split2event[self.mode]:
            if self.mode == 'train':
                aux_events = list(set(self.split2event['train']) - {event})
            else:
                aux_events = list(set(self.split2event['train']))
            tgt_event_idx = self.args.event2idx[event]
            aux_events_idx = [self.args.event2idx[_] for _ in aux_events]
            tgt_sg = get_subgraph_by_rels(self.g, tgt_event_idx)
            aux_sg = get_subgraph_by_rels(self.g, aux_events_idx)

            if self.args.binary:
                tgt_sg.edata['etype'] = torch.tensor([1] * len(tgt_sg.edges()[0]), dtype=torch.long)
                aux_sg.edata['etype'] = torch.tensor([2] * len(aux_sg.edges()[0]), dtype=torch.long)
            else:
                if self.args.tgt_flag:
                    tgt_index = self.args.num_rels + 1
                    tgt_sg.edata['etype'] = torch.tensor([tgt_index] * len(tgt_sg.edges()[0]), dtype=torch.long)

            event2tgt_sg.update({event: tgt_sg})
            event2aux_sg.update({event: aux_sg})
        return event2tgt_sg, event2aux_sg

    def sample(self):
        self.input_graphs = []
        self.tgt_graphs = []
        self.support_edges = []
        self.valid_edges = []
        self.query_edges = []
        self.tgt_events = []
        self.evaluate_edges = []

        for idx in range(self.num_sample_tasks):
            candidate_events = self.split2event[self.mode]
            if self.mode == 'train':
                tgt_event = np.random.choice(candidate_events, size=1, replace=False)[0]
            else:  # ['valid', 'test']:
                tgt_event = candidate_events[idx % len(candidate_events)]

            tgt_sg = self.event2tgt_sg[tgt_event]
            aux_sg = self.event2aux_sg[tgt_event]
            assert tgt_sg.number_of_nodes() == aux_sg.number_of_nodes() == self.args.num_nodes

            if self.mode == 'train':
                tgt_masked = add_masks(g=tgt_sg, train_ratio=self.args.train_ratio, valid_ratio=0)
                tgt_support_edges = get_subgraph_by_mask(g=tgt_masked, mask=tgt_masked.edata['support_mask'],
                                                         to_bidirect=False)
                tgt_query_edges = get_subgraph_by_mask(g=tgt_masked, mask=tgt_masked.edata['query_mask'],
                                                       to_bidirect=False)

                if self.args.type == 'split':
                    support_triplets, support_labels = sample_edges_split(tgt_edges=tgt_support_edges, aux_edges=aux_sg,
                                                                          num_tgt_edges=self.args.tgt_bsz,
                                                                          num_aux_edges=self.args.aux_bsz)
                    query_triplets, query_labels = sample_edges_split(tgt_edges=tgt_query_edges, aux_edges=aux_sg,
                                                                      num_tgt_edges=self.args.tgt_bsz,
                                                                      num_aux_edges=self.args.aux_bsz)
                else:
                    support_triplets, support_labels = sample_edges_direct(tgt_edges=tgt_support_edges,
                                                                           aux_edges=aux_sg,
                                                                           bsz=self.args.bsz)
                    query_triplets, query_labels = sample_edges_direct(tgt_edges=tgt_query_edges, aux_edges=aux_sg,
                                                                       bsz=self.args.bsz)

                self.support_edges.append((support_triplets, support_labels))  # [edges, labels]
                self.query_edges.append((query_triplets, query_labels))

            elif self.mode in ['valid', 'test']:
                tgt_masked = add_masks(g=tgt_sg, train_ratio=self.args.train_ratio, valid_ratio=self.args.valid_ratio)
                tgt_support_edges = get_subgraph_by_mask(g=tgt_masked, mask=tgt_masked.edata['support_mask'],
                                                         to_bidirect=False)
                tgt_valid_edges = get_subgraph_by_mask(g=tgt_masked, mask=tgt_masked.edata['valid_mask'],
                                                       to_bidirect=False)
                tgt_query_edges = get_subgraph_by_mask(g=tgt_masked, mask=tgt_masked.edata['query_mask'],
                                                       to_bidirect=False)
                support_triplets = edges2triplets(tgt_graph=tgt_support_edges, aux_graph=aux_sg)
                valid_triplets = edges2triplets(tgt_graph=tgt_valid_edges, aux_graph=aux_sg)
                query_triplets = edges2triplets(tgt_graph=tgt_query_edges, aux_graph=aux_sg)
                self.support_edges.append(support_triplets)
                self.valid_edges.append(valid_triplets)
                self.query_edges.append(query_triplets)
            else:
                raise NotImplementedError

            aux_ratio = self.args.aux_ratio if self.mode == 'train' else 1
            input_graph = merge_graphs(tgt_graph=tgt_support_edges, aux_graph=aux_sg, to_bidirect=True, aux_ratio=aux_ratio)
            input_graph = dgl.remove_self_loop(input_graph)
            input_graph = dgl.add_self_loop(input_graph)

            self.input_graphs.append(input_graph)
            self.tgt_graphs.append(tgt_support_edges)
            self.tgt_events.append(tgt_event)

    def __getitem__(self, idx):
        if self.mode in ['valid', 'test']:
            return self.input_graphs[idx], self.tgt_graphs[idx], self.support_edges[idx], \
                   self.valid_edges[idx], self.query_edges[idx], self.tgt_events[idx]
        elif self.mode == 'train':
            return self.input_graphs[idx], self.tgt_graphs[idx], self.support_edges[idx], \
                   self.query_edges[idx], self.tgt_events[idx]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.num_sample_tasks


def get_pos_neg_triplets(args, mode, triplets, tgt_event_idx):
    tgt_mask = (triplets[:, 1] == tgt_event_idx)
    tgt_triplets, aux_triplets = triplets[tgt_mask], triplets[tgt_mask == False]

    if args.type == 'split':
        if mode == 'support':
            chosen_tgt_pos_idx = np.random.choice(len(tgt_triplets), args.tgt_bsz)
            tgt_pos_triplets = tgt_triplets[chosen_tgt_pos_idx]
        elif mode in ['valid', 'query']:
            tgt_pos_triplets = tgt_triplets
        else:
            raise NotImplementedError
        chosen_aux_pos_idx = np.random.choice(len(aux_triplets), args.aux_bsz)
        aux_pos_triplets = aux_triplets[chosen_aux_pos_idx]

        tgt_triplets, tgt_labels = negative_sampling(pos_samples=tgt_pos_triplets, num_nodes=args.num_nodes, k=1)
        aux_triplets, aux_labels = negative_sampling(pos_samples=aux_pos_triplets, num_nodes=args.num_nodes, k=1)

        triplets = np.concatenate((tgt_triplets, aux_triplets))
        labels = np.concatenate((tgt_labels, aux_labels))

    else:
        if mode == 'support':
            chosen_pos_idx = np.random.choice(len(triplets), args.bsz)
            chosen_pos_triplets = triplets[chosen_pos_idx]
        elif mode in ['valid', 'query']:
            chosen_pos_triplets = triplets
        else:
            raise NotImplementedError

        triplets, labels = negative_sampling(pos_samples=chosen_pos_triplets, num_nodes=args.num_nodes, k=1)

    triplets = torch.tensor(triplets, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    return triplets, labels


def get_subgraph_by_mask(g, mask, to_bidirect=False):
    src, dst = g.edges()
    sub_src = src[mask]
    sub_dst = dst[mask]
    sub_rel = g.edata['etype'][mask]

    if to_bidirect:
        sub_src, sub_dst = torch.cat([sub_src, sub_dst]), torch.cat([sub_dst, sub_src])
        sub_rel = torch.cat([sub_rel, sub_rel])

    sub_g = dgl.graph((sub_src, sub_dst), num_nodes=g.num_nodes())
    sub_g.edata['etype'] = sub_rel

    return sub_g


def get_subgraph_by_rels(g, candidate_rels):
    if not isinstance(candidate_rels, list):
        candidate_rels = [candidate_rels]
    src, dst = g.edges()
    rels = g.edata['etype']

    mask = [_ in candidate_rels for _ in rels]
    sub_src = src[mask]
    sub_dst = dst[mask]
    sub_rel = g.edata['etype'][mask]

    sub_g = dgl.graph((sub_src, sub_dst), num_nodes=g.num_nodes())
    sub_g.edata['etype'] = sub_rel

    return sub_g


def add_masks(g, train_ratio, valid_ratio=0):
    etypes = g.edata['etype'].tolist()
    assert len(set(etypes)) == 1

    edge_type = etypes[0]

    src, dst = g.edges()
    perm = torch.randperm(len(src))
    src = src[perm]
    dst = dst[perm]
    edges = list(zip(src.tolist(), dst.tolist()))

    src_list = []
    dst_list = []
    rel_list = []
    train_mask = []
    valid_mask = []
    test_mask = []

    train_idx = round(len(edges) * train_ratio)
    valid_idx = round(len(edges) * (train_ratio + valid_ratio))

    train_edges = edges[: train_idx]
    valid_edges = edges[train_idx: valid_idx]
    test_edges = edges[valid_idx:]

    for src, dst in train_edges:
        src_list.append(src)
        dst_list.append(dst)
        rel_list.append(edge_type)
        train_mask.append(1)
        valid_mask.append(0)
        test_mask.append(0)

    for src, dst in valid_edges:
        src_list.append(src)
        dst_list.append(dst)
        rel_list.append(edge_type)
        train_mask.append(0)
        valid_mask.append(1)
        test_mask.append(0)

    for src, dst in test_edges:
        src_list.append(src)
        dst_list.append(dst)
        rel_list.append(edge_type)
        train_mask.append(0)
        valid_mask.append(0)
        test_mask.append(1)

    src_list = torch.tensor(src_list, dtype=torch.long)
    dst_list = torch.tensor(dst_list, dtype=torch.long)
    rel_list = torch.tensor(rel_list, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    g = dgl.graph((src_list, dst_list), num_nodes=g.number_of_nodes())
    g.edata['support_mask'] = train_mask
    g.edata['valid_mask'] = valid_mask
    g.edata['query_mask'] = test_mask
    g.edata['etype'] = rel_list

    return g


def global_uniform(g, sample_size=None):
    if sample_size is None:
        src, dst = g.edges()
        rel = g.edata['etype']
    else:
        eids = np.arange(g.num_edges())
        chosen_eids = torch.from_numpy(np.random.choice(eids, sample_size))
        src, dst = g.find_edges(chosen_eids)
        rel = g.edata['etype'][chosen_eids]

    src, rel, dst = src.numpy(), rel.numpy(), dst.numpy()
    pos_triplets = np.stack((src, rel, dst)).transpose()
    return pos_triplets


def negative_sampling(pos_samples, num_nodes, k=1):
    batch_size = len(pos_samples)
    neg_batch_size = batch_size * k
    neg_samples = np.tile(pos_samples, (k, 1))

    values = np.random.randint(num_nodes, size=neg_batch_size)
    choices = np.random.uniform(size=neg_batch_size)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]
    samples = np.concatenate((pos_samples, neg_samples))

    # binary labels indicating positive and negative samples
    labels = np.zeros(batch_size * (k + 1), dtype=np.float32)
    labels[:batch_size] = 1  # 1
    labels[batch_size:] = 0  # 0

    samples = torch.from_numpy(samples)
    labels = torch.from_numpy(labels)

    return samples, labels


def edges2triplets(tgt_graph, aux_graph):
    assert tgt_graph.number_of_nodes() == aux_graph.number_of_nodes()
    tgt_src, tgt_dst = tgt_graph.edges()
    tgt_rel = tgt_graph.edata['etype']
    aux_src, aux_dst = aux_graph.edges()
    aux_rel = aux_graph.edata['etype']

    src = torch.cat((tgt_src, aux_src)).numpy()
    rel = torch.cat((tgt_rel, aux_rel)).numpy()
    dst = torch.cat((tgt_dst, aux_dst)).numpy()
    triplets = np.stack((src, rel, dst)).transpose()

    return triplets


def merge_graphs(tgt_graph, aux_graph, to_bidirect, aux_ratio=1):
    assert tgt_graph.number_of_nodes() == aux_graph.number_of_nodes()
    num_nodes = tgt_graph.number_of_nodes()

    tgt_src, tgt_dst = tgt_graph.edges()
    tgt_rel = tgt_graph.edata['etype']

    aux_src, aux_dst = aux_graph.edges()
    aux_rel = aux_graph.edata['etype']
    if aux_ratio<1:
        eids = np.arange(len(aux_src))
        chosen_eids = torch.from_numpy(np.random.choice(eids, math.ceil(aux_ratio*len(eids))))
        aux_src, aux_dst, aux_rel = aux_src[chosen_eids], aux_dst[chosen_eids], aux_rel[chosen_eids]

    src_list = tgt_src.tolist() + aux_src.tolist()
    dst_list = tgt_dst.tolist() + aux_dst.tolist()
    rel_list = tgt_rel.tolist() + aux_rel.tolist()

    src_list = torch.tensor(src_list, dtype=torch.long)
    dst_list = torch.tensor(dst_list, dtype=torch.long)
    rel_list = torch.tensor(rel_list, dtype=torch.long)

    if to_bidirect:
        src_list, dst_list = torch.cat([src_list, dst_list]), torch.cat([dst_list, src_list])
        rel_list = torch.cat([rel_list, rel_list])  #

    g = dgl.graph((src_list, dst_list), num_nodes=num_nodes)
    g.edata['etype'] = rel_list

    return g


def sample_edges_split(tgt_edges, aux_edges, num_tgt_edges, num_aux_edges):
    num_nodes = tgt_edges.number_of_nodes()
    tgt_pos_triplets = global_uniform(g=tgt_edges, sample_size=num_tgt_edges)
    tgt_triplets, tgt_labels = negative_sampling(pos_samples=tgt_pos_triplets, num_nodes=num_nodes, k=1)
    aux_pos_triplets = global_uniform(g=aux_edges, sample_size=num_aux_edges)
    aux_triplets, aux_labels = negative_sampling(pos_samples=aux_pos_triplets, num_nodes=num_nodes, k=1)

    triplets = np.concatenate((tgt_triplets, aux_triplets))
    labels = np.concatenate((tgt_labels, aux_labels))

    triplets = torch.tensor(triplets, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)

    return triplets, labels


def sample_edges_direct(tgt_edges, aux_edges, bsz):
    edges = merge_graphs(tgt_graph=tgt_edges, aux_graph=aux_edges, to_bidirect=False)
    num_nodes = edges.number_of_nodes()
    pos_triplets = global_uniform(g=edges, sample_size=bsz)
    triplets, labels = negative_sampling(pos_samples=pos_triplets, num_nodes=num_nodes, k=1)
    triplets = torch.tensor(triplets, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)

    return triplets, labels
