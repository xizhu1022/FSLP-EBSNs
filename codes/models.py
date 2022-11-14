import math

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, precision_score

from conv import myGATConv, GATConv


class EncoderDecoder(nn.Module):
    def __init__(self,
                 args,
                 dim,
                 num_nodes,
                 num_rels,
                 num_layers,
                 reg_param,
                 aux_param,
                 pretrained_node_feats
                 ):
        super(EncoderDecoder, self).__init__()
        self.args = args
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_layers = num_layers
        self.reg_param = reg_param
        self.aux_param = aux_param

        if pretrained_node_feats is None:
            self.node_feats = nn.Parameter(torch.zeros((self.num_nodes, self.dim)), requires_grad=True)
        else:
            self.node_feats = nn.Parameter(pretrained_node_feats, requires_grad=True)

        if self.args.encoder == 'gat':
            self.encoder = GAT(args, edge_feats=self.dim, num_etypes=self.num_rels, in_dim=self.dim,
                               hidden_dim=self.dim, num_layers=self.args.num_layers, num_heads=self.args.num_heads,
                               activation=F.relu, feat_drop=self.args.dropout, attn_drop=self.args.dropout,
                               negative_slope=0.2, residual=False, alpha=0.)
        elif self.args.encoder == 'hgb':
            self.encoder = myGAT(args, edge_feats=self.dim, num_etypes=self.num_rels, in_dim=self.dim,
                                 hidden_dim=self.dim, num_layers=self.args.num_layers, num_heads=self.args.num_heads,
                                 activation=F.elu, feat_drop=self.args.dropout, attn_drop=self.args.dropout,
                                 negative_slope=0.2, residual=True, alpha=0.05)


        if self.args.decoder in ['rel','distmult']:
            if self.args.encoder == 'hgb':
                decoder_dim = self.dim if self.args.pooling in ['mean','last'] else self.dim * (self.num_layers + 2)
            elif self.args.encoder == 'gat':
                decoder_dim = self.dim * self.args.num_heads
            else:
                raise NotImplementedError
            if self.args.decoder == 'rel':
                self.decoder = RelDecoder(num_rels=self.num_rels, dim=decoder_dim)
            elif self.args.decoder == 'distmult':
                self.decoder = DistMult(num_rels=self.num_rels, dim=decoder_dim)
            else:
                raise NotImplementedError

        elif self.args.decoder == 'dot':
            self.decoder = Dot()

        elif self.args.decoder == 'mlp':
            if self.args.encoder == 'gat':
                self.decoder = MLP(in_dim=self.dim * self.args.num_heads * 2, dim=self.dim)
            elif self.args.encoder == 'hgb':
                decoder_dim = self.dim if self.args.pooling in ['mean','last'] else self.dim * (self.num_layers + 2)
                self.decoder = MLP(in_dim=decoder_dim * 2, dim=self.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.node_feats.data)

    def forward(self, input_graphs, feature_graphs, triplets, labels, tgt_event, vars):
        if vars is None:
            vars = self.named_parameters()

        h = self.encoder.forward(g=input_graphs, e_feat=input_graphs.edata['etype'],
                                 node_feats=vars['node_feats'],
                                 vars=vars)

        scores = self.decoder.forward(triplets=triplets, node_emb=h, vars=vars)

        return self.compute_loss(tgt_event=tgt_event, triplets=triplets, scores=scores, labels=labels, vars=vars)

    def regularization_loss(self, vars):
        node_embs = vars['node_feats']
        if 'decoder.W' in vars.keys():
            assert self.args.decoder in ['distmult', 'rel']
            rel_embs = vars['decoder.W']
            loss = torch.mean(node_embs.pow(2)) + torch.mean(rel_embs.pow(2))
        else:
            loss = torch.mean(node_embs.pow(2))
        return loss

    def compute_loss(self, tgt_event, triplets, scores, labels, vars):

        if self.args.binary:
            tgt_event_idx = 1
        else:
            if self.args.tgt_flag:
                tgt_event_idx = self.args.num_rels + 1
            else:
                tgt_event_idx = self.args.event2idx[tgt_event]

        tgt_mask = (triplets[:, 1] == tgt_event_idx)
        tgt_scores, aux_scores = scores[tgt_mask], scores[tgt_mask == False]
        tgt_labels, aux_labels = labels[tgt_mask], labels[tgt_mask == False]

        tgt_predict_loss = F.binary_cross_entropy_with_logits(tgt_scores, tgt_labels, reduction='sum')
        aux_predict_loss = F.binary_cross_entropy_with_logits(aux_scores, aux_labels, reduction='sum')

        tgt_loss = tgt_predict_loss / len(tgt_scores)
        aux_loss = aux_predict_loss / len(aux_scores)
        reg_loss = self.reg_param * self.regularization_loss(vars=vars)

        loss = (tgt_predict_loss + self.aux_param * aux_predict_loss) / len(triplets) + reg_loss

        loss_dict = {'loss': loss.detach().cpu().item(), 'tgt_loss': tgt_loss.detach().cpu().item(),
                     'aux_loss': aux_loss.detach().cpu().item(), 'reg_loss': reg_loss.detach().cpu().item()}
        tgt_scores, tgt_labels = tgt_scores.detach().cpu(), tgt_labels.detach().cpu()
        return loss, loss_dict, tgt_scores, tgt_labels

    def evaluate(self, scores, labels):
        pred = (F.sigmoid(scores) > 0.5)
        labels, scores, pred = labels.numpy(), scores.numpy(), pred.numpy()
        assert np.sum(labels == 0) == np.sum(labels == 1)

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        f1 = f1_score(labels, pred)
        p = precision_score(y_true=labels, y_pred=pred)
        r = recall_score(y_true=labels, y_pred=pred)
        acc = accuracy_score(y_true=labels, y_pred=pred)
        metrics = {'auc': auc, 'f1': f1, 'ap': ap, 'p': p, 'r': r, 'acc': acc}
        return metrics


class RelDecoder(nn.Module):
    def __init__(self, num_rels, dim):
        super(RelDecoder, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rels, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, triplets, node_emb, vars):
        left, mid, right = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        left_emb = node_emb[torch.tensor(left, dtype=torch.long)]
        right_emb = node_emb[torch.tensor(right, dtype=torch.long)]
        rel_emb = vars['decoder.W'][torch.tensor(mid, dtype=torch.long)]
        scores = torch.sum(left_emb * rel_emb * right_emb, dim=1)
        return scores


class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, triplets, node_emb, vars):
        left, mid, right = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        left_emb = node_emb[torch.tensor(left, dtype=torch.long)].unsqueeze_(1)
        right_emb = node_emb[torch.tensor(right, dtype=torch.long)].unsqueeze_(2)
        scores = torch.bmm(left_emb, right_emb).squeeze()
        return scores


class MLP(nn.Module):
    def __init__(self, in_dim, dim):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(dim, 1)
        self.activation = nn.ReLU()

    def forward(self, triplets, node_emb, vars):
        left, mid, right = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        left_emb = node_emb[torch.tensor(left, dtype=torch.long)]
        right_emb = node_emb[torch.tensor(right, dtype=torch.long)]
        h = F.linear(torch.cat((left_emb, right_emb), dim=-1), weight=vars['decoder.w1.weight'],
                     bias=vars['decoder.w1.bias'])
        h = self.activation(h)

        score = F.linear(h, weight=vars['decoder.w2.weight'], bias=vars['decoder.w2.bias'])
        score = score.squeeze()
        return score


class DistMult(nn.Module):
    def __init__(self, num_rels, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rels, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def get_scores_with_g(self, triplets, node_emb, vars):
        src, rel, dst = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        g = dgl.graph((src, dst), num_nodes=node_emb.size(0))

        with g.local_scope():
            g.ndata['h'] = node_emb
            g.edata['etype'] = torch.tensor(rel, dtype=torch.long)
            g.apply_edges(
                lambda edges: {
                    'h_e': torch.sum(edges.src['h'].unsqueeze(1) * vars['decoder.W'][edges.data['etype']], dim=1)}
            )
            g.apply_edges(fn.e_dot_v('h_e', 'h', 'score'))
            scores = g.edata['score'][:, 0]
        return scores

    def get_scores_direct(self, triplets, node_emb, vars):
        src, rel, dst = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        src_emb, dst_emb = node_emb[src], node_emb[dst]
        src_emb = torch.unsqueeze(src_emb, dim=1)
        dst_emb = torch.unsqueeze(dst_emb, dim=2)
        rel_emb = vars['decoder.W'][rel]
        scores = torch.bmm(torch.bmm(src_emb, rel_emb), dst_emb).squeeze()
        return scores

    def forward(self, triplets, node_emb, vars):
        batch_size = 100000
        all_scores = []

        num_triplets = len(triplets)
        num_batches = math.ceil(num_triplets / batch_size)

        for i in range(num_batches):
            start = i * batch_size
            end = min(num_triplets, (i + 1) * batch_size)
            batch_triplets = triplets[start:end]
            batch_scores = self.get_scores_direct(batch_triplets, node_emb, vars)
            # batch_scores = self.get_scores_with_g(batch_triplets, node_emb, vars)
            all_scores.append(batch_scores)
        all_scores = torch.cat(all_scores)
        return all_scores


class myGAT(nn.Module):
    def __init__(self,
                 args,
                 edge_feats,
                 num_etypes,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 num_heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha
                 ):
        super(myGAT, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.epsilon = torch.FloatTensor([1e-12]).to(self.args.device)

        self.gat_layers = nn.ModuleList()

        # input projection (no residual, with activation)
        self.gat_layers.append(
            myGATConv(edge_feats=edge_feats, num_etypes=num_etypes, in_feats=in_dim, out_feats=hidden_dim,
                      num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                      residual=False, activation=self.activation, allow_zero_in_degree=False, bias=False, alpha=alpha)
        )

        # hidden layers (with residual and activation)
        for l in range(1, self.num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                myGATConv(edge_feats=edge_feats, num_etypes=num_etypes, in_feats=hidden_dim * num_heads,
                          out_feats=hidden_dim, num_heads=num_heads,feat_drop=feat_drop, attn_drop=attn_drop,
                          negative_slope=negative_slope, residual=residual, activation=self.activation,
                          allow_zero_in_degree=False, bias=False, alpha=alpha)
            )

        # output projection (with residual, no activation)
        self.gat_layers.append(
            myGATConv(edge_feats=edge_feats, num_etypes=num_etypes, in_feats=hidden_dim * num_heads,
                      out_feats=hidden_dim, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                      negative_slope=negative_slope, residual=residual, activation=None, allow_zero_in_degree=False,
                      bias=False, alpha=alpha)
        )

    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize,
        # see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return x / torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon)

    def forward(self, g, e_feat, node_feats, vars):
        """
        :param g: The graph, DGLGraph
        :param node_feats: Node feature, torch.Tensor or list of torch.Tensor
        :param e_feat: Types of edges
        :return: logits: Node feature
        """
        if node_feats is None:
            node_feats = vars['node_feats']
        h = node_feats
        node_embs = [self.l2_norm(h)]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g=g, feat=h, e_feat=e_feat, layer_idx=l, res_attn=res_attn, vars=vars)
            # h:(num_nodes, heads[l], hidden_dim)
            node_embs.append(self.l2_norm(torch.mean(h, dim=1)))
            # (num_nodes, hidden_dim)
            h = h.flatten(1)
            # (num_nodes, heads[l] * hidden_dim)
        # output projection
        logits, _ = self.gat_layers[-1](g=g, feat=h, e_feat=e_feat, layer_idx=self.num_layers,
                                        res_attn=res_attn, vars=vars)
        logits = torch.mean(logits, dim=1)  # (num_nodes, hidden_dim)
        logits = self.l2_norm(logits)
        node_embs.append(logits)
        if self.args.pooling == 'cat':
            logits = torch.cat(node_embs, dim=1)  # (num_nodes, (num_layers+1) * hidden_dim)
            assert logits.shape[-1] == self.hidden_dim * (self.num_layers+2)
            return logits
        elif self.args.pooling == 'mean':
            logits = torch.mean(torch.stack(node_embs, dim=0), dim=0)  # (num_nodes, hidden_dim)
            assert logits.shape[-1] == self.hidden_dim
            return logits
        elif self.args.pooling == 'last':
            assert logits.shape[-1] == self.hidden_dim
            return logits
        else:
            raise NotImplementedError

class GAT(nn.Module):
    def __init__(self,
                 args,
                 edge_feats,
                 num_etypes,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 num_heads,
                 activation,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 alpha=0.
                 ):
        super(GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.activation = activation
        self.epsilon = torch.FloatTensor([1e-12]).to(self.args.device)
        self.gat_layers = nn.ModuleList()

        # input projection (no residual, with activation)
        self.gat_layers.append(
            GATConv(edge_feats=edge_feats, num_etypes=num_etypes, in_feats=in_dim, out_feats=hidden_dim,
                    num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                    negative_slope=negative_slope, residual=False, activation=self.activation,
                    allow_zero_in_degree=False, bias=True, alpha=alpha)
        )

        # hidden layers (with residual and activation)
        for l in range(1, self.num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(edge_feats=edge_feats, num_etypes=num_etypes, in_feats=hidden_dim * num_heads,
                        out_feats=hidden_dim, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                        negative_slope=negative_slope, residual=residual, activation=self.activation,
                        allow_zero_in_degree=False, bias=True, alpha=alpha)
            )

        # output projection (with residual, no activation)
        self.gat_layers.append(
            GATConv(edge_feats=edge_feats, num_etypes=num_etypes, in_feats=hidden_dim * num_heads,
                    out_feats=hidden_dim, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                    negative_slope=negative_slope, residual=residual, activation=None, allow_zero_in_degree=False,
                    bias=True, alpha=alpha)
        )

    def l2_norm(self, x):
        return x / torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon)

    def forward(self, g, e_feat, node_feats, vars):
        """
        :param g: The graph, DGLGraph
        :param node_feats: Node feature, torch.Tensor or list of torch.Tensor
        :param e_feat: Types of edges
        :return: logits: Node feature
        """
        if node_feats is None:
            node_feats = vars['node_feats']

        h = node_feats
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g=g, feat=h, e_feat=e_feat, layer_idx=l, res_attn=res_attn, vars=vars)
            # h:(num_nodes, heads[l], hidden_dim)
            h = torch.reshape(h, (h.shape[0], -1))
            # (num_nodes, heads[l] * hidden_dim)

        # output projection
        h, _ = self.gat_layers[-1](g=g, feat=h, e_feat=e_feat, layer_idx=self.num_layers, res_attn=res_attn, vars=vars)
        h = torch.reshape(h, (h.shape[0], -1))  # (num_nodes, hidden_dim *head)

        return h
