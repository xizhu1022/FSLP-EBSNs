
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl._ffi.base import DGLError
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair



class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    Copied from HGN paper
    https://github.com/THUDM/HGB/blob/d68321c4e1568813e1c386be39f9705eedd68ef4/LP/benchmark/methods/baseline/conv.py
    """

    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 activation,
                 allow_zero_in_degree,
                 bias,
                 alpha
                 ):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Parameter(torch.FloatTensor(size=(num_etypes, self._edge_feats)))  # Edge Embedding

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * self._num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * self._num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * self._num_heads, bias=False)

        self.fc_e = nn.Linear(edge_feats, edge_feats * self._num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        if residual:
            if self._in_dst_feats != self._out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.edge_emb.data, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, g, feat, e_feat, layer_idx, res_attn=None, vars=None):
        '''
         g: graph
         feat: node features
         e_feat, edge feature index
         layer_idx: the index of GATConv layer
        '''
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = feat_dst = \
                        F.linear(h_src, weight=vars['encoder.gat_layers.{}.fc.weight'.format(layer_idx)]) \
                            .view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = F.linear(h_src, weight=vars['encoder.gat_layers.{}.fc_src.weight'.format(layer_idx)]) \
                        .view(-1, self._num_heads, self._out_feats)
                    feat_dst = F.linear(h_dst, weight=vars['encoder.gat_layers.{}.fc_dst.weight'.format(layer_idx)]) \
                        .view(-1, self._num_heads, self._out_feats)  # (num_nodes, num_heads, out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)  # (num_nodes, in_src_feats)
                feat_src = feat_dst = F.linear(h_src, weight=vars['encoder.gat_layers.{}.fc.weight'.format(layer_idx)]) \
                    .view(-1, self._num_heads, self._out_feats)  # (num_nodes, num_heads, out_feats)
                if g.is_block:
                    feat_dst = feat_src[:g.number_of_dst_nodes()]

            e_feat = vars['encoder.gat_layers.{}.edge_emb'.format(layer_idx)][torch.LongTensor(e_feat.cpu())]
            e_feat = F.linear(e_feat, weight=vars['encoder.gat_layers.{}.fc_e.weight'.format(layer_idx)]). \
                view(-1, self._num_heads, self._edge_feats)  # (num_edges, num_heads, edge_feats)

            ee = (e_feat * vars['encoder.gat_layers.{}.attn_e'.format(layer_idx)]).sum(dim=-1).unsqueeze(
                -1)  # (num_edges, num_heads, 1)
            el = (feat_src * vars['encoder.gat_layers.{}.attn_l'.format(layer_idx)]).sum(dim=-1).unsqueeze(
                -1)  # (num_nodes, num_heads, 1)
            er = (feat_dst * vars['encoder.gat_layers.{}.attn_r'.format(layer_idx)]).sum(dim=-1).unsqueeze(
                -1)  # (num_nodes, num_heads, 1)

            g.srcdata.update({'ft': feat_src, 'el': el})
            # ft: (num_nodes, num_heads, out_feats), el: (num_nodes, num_heads, 1)

            g.dstdata.update({'er': er})  # er: (num_nodes, num_heads, 1)
            g.edata.update({'ee': ee})  # ee: (num_edges, num_heads, 1)
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))  # e: (num_edges, num_heads, 1)
            e = self.leaky_relu(g.edata.pop('e') + g.edata.pop('ee'))  # e: (num_edges, num_heads, 1)

            g.edata['a'] = self.attn_drop(edge_softmax(g, e))  # a: (num_edges, num_heads, 1)
            if res_attn is not None:
                g.edata['a'] = g.edata['a'] * (1 - self.alpha) + res_attn * self.alpha

            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

            rst = g.dstdata['ft']  # (num_nodes, num_heads, out_feats)

            # residual
            if self.res_fc is not None:
                if self._in_dst_feats != self._out_feats:
                    resval = F.linear(h_dst, weight=vars['encoder.gat_layers.{}.res_fc.weight'.format(layer_idx)]). \
                        view(h_dst.shape[0], -1, self._out_feats)  # (num_nodes, num_heads, out_feats)
                    rst = rst + resval
                else:
                    resval = h_dst.view(h_dst.shape[0], -1, self._out_feats)  # (num_nodes, num_heads, out_feats)
                    rst = rst + resval

            # bias
            if self.bias:
                rst = rst + vars['encoder.gat_layers.{}.bias_param'.format(layer_idx)]
            # activation
            if self.activation:
                rst = self.activation(rst)  # (num_nodes, num_heads, out_feats)
            attn = g.edata.pop('a').detach()  # (num_edges, num_heads, 1)
            return rst, attn


class GATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    Copied from HGN paper
    https://github.com/THUDM/HGB/blob/d68321c4e1568813e1c386be39f9705eedd68ef4/LP/benchmark/methods/baseline/conv.py
    """

    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 alpha=0.
                 ):
        super(GATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * self._num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * self._num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * self._num_heads, bias=False)

        # self.fc_e = nn.Linear(edge_feats, edge_feats * self._num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats)))

        # Dropout
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        if residual:
            if self._in_dst_feats != self._out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, g, feat, e_feat, layer_idx, res_attn=None, vars=None):
        '''
         g: graph
         feat: node features
         e_feat, edge feature index
         layer_idx: the index of GATConv layer
        '''
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = feat_dst = \
                        F.linear(h_src, weight=vars['encoder.gat_layers.{}.fc.weight'.format(layer_idx)]) \
                            .view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = F.linear(h_src, weight=vars['encoder.gat_layers.{}.fc_src.weight'.format(layer_idx)]) \
                        .view(-1, self._num_heads, self._out_feats)
                    feat_dst = F.linear(h_dst, weight=vars['encoder.gat_layers.{}.fc_dst.weight'.format(layer_idx)]) \
                        .view(-1, self._num_heads, self._out_feats)  # (num_nodes, num_heads, out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)  # (num_nodes, in_src_feats)
                feat_src = feat_dst = F.linear(h_src, weight=vars['encoder.gat_layers.{}.fc.weight'.format(layer_idx)]) \
                    .view(-1, self._num_heads, self._out_feats)  # (num_nodes, num_heads, out_feats)
                if g.is_block:
                    feat_dst = feat_src[:g.number_of_dst_nodes()]

            el = (feat_src * vars['encoder.gat_layers.{}.attn_l'.format(layer_idx)]).sum(dim=-1).unsqueeze(
                -1)  # (num_nodes, num_heads, 1)
            er = (feat_dst * vars['encoder.gat_layers.{}.attn_r'.format(layer_idx)]).sum(dim=-1).unsqueeze(
                -1)  # (num_nodes, num_heads, 1)

            g.srcdata.update({'ft': feat_src, 'el': el}) # ft: (num_nodes, num_heads, out_feats), el: (num_nodes, num_heads, 1)

            g.dstdata.update({'er': er})  # er: (num_nodes, num_heads, 1)

            g.apply_edges(fn.u_add_v('el', 'er', 'e'))  # e: (num_edges, num_heads, 1)
            e = self.leaky_relu(g.edata.pop('e') )  # e: (num_edges, num_heads, 1)
            g.edata['a'] = self.attn_drop(edge_softmax(g, e))  # a: (num_edges, num_heads, 1)
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = g.dstdata['ft']  # (num_nodes, num_heads, out_feats)

            # residual
            if self.res_fc is not None:
                if self._in_dst_feats != self._out_feats:
                    resval = F.linear(h_dst, weight=vars['encoder.gat_layers.{}.res_fc.weight'.format(layer_idx)]). \
                        view(h_dst.shape[0], -1, self._out_feats)  # (num_nodes, num_heads, out_feats)
                    rst = rst + resval
                else:
                    resval = h_dst.view(h_dst.shape[0], -1, self._out_feats)  # (num_nodes, num_heads, out_feats)
                    rst = rst + resval

            # bias
            if self.bias:
                rst = rst + vars['encoder.gat_layers.{}.bias_param'.format(layer_idx)]
            # activation
            if self.activation:
                rst = self.activation(rst)  # (num_nodes, num_heads, out_feats)
            attn = g.edata.pop('a').detach()  # (num_edges, num_heads, 1)
            return rst, attn




