import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

# MLP
import time
import math
from torch.nn.parameter import Parameter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import dgl


def plot_encoded_eigenvalues(ax, encoded, epsilon,fig):
    cax = ax.imshow(encoded.T, cmap='coolwarm', aspect='auto')
    ax.set_title(f"ε = {epsilon}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Eigenvalues")
    fig.colorbar(cax, ax=ax)
class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000) / self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class SpecLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none':
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':  # Arxiv
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'batch':  # Penn
            self.norm = nn.BatchNorm1d(ncombines)
        else:  # Others
            self.norm = None

    def forward(self, x):
        x = self.prop_dropout(x) * self.weight  # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x
class Ortho_Trans(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(Ortho_Trans, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)

        S = S.div(norm_S)
        B = [torch.FloatTensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        return W.view_as(weight)

class Ortho_GCNII(nn.Module):
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    def __init__(self, in_features, out_features, T2, group2, weight_beta, Ortho, residual=False, variant=False):
        super(Ortho_GCNII, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.Ortho = Ortho
        self.weight_beta = weight_beta
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight_normalization = Ortho_Trans(T=2, norm_groups=2)  # 2，2 2,1            cora(5141)
        self.bn = torch.nn.BatchNorm1d(in_features)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()
        self.t = None
        self.we = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support

        if self.Ortho:
            I = torch.eye(self.in_features)
            self.we = self.weight_beta * self.weight + (1 - self.weight_beta) * I
            self.t = self.weight_normalization(self.we)

            if self.training:
                self.t.retain_grad()
            output = theta * torch.mm(support, self.t) + (1 - theta) * r
        else:
            output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + 0.05 * input
        return output
class Specformer(nn.Module):

    def __init__(self,adj,adjmax,transposed_list,e,u, G, hid_dim, n_class, S, K, batchnorm, num_diseases, num_mirnas,
                 d_sim_dim, m_sim_dim, out_dim, dropout, slope, node_dropout=0.5, input_droprate=0.0,
                 hidden_droprate=0.0,nclass=512, nfeat=512, nlayer=1, hidden_dim=512, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):

        super(Specformer, self).__init__()

        self.adjmax = adjmax
        self.adj = adj
        self.transposed_list = transposed_list
        self.e = e
        self.u = u
        self.G = G
        self.hid_dim = hid_dim
        self.S = S
        self.K = K
        self.n_class = n_class
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.baixu = nn.Linear(512,64)

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], hid_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], hid_dim, bias=False)
        self.m_fc1 = nn.Linear(64, out_dim)
        self.d_fc1 = nn.Linear( 64, out_dim)
        self.B = nn.Linear(64, 64)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = MLP(hid_dim, out_dim, n_class, input_droprate, hidden_droprate, batchnorm)




        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)
        self.predict = nn.Linear(out_dim * 2, 1)

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # for arxiv & penn
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        self.classify = nn.Linear(hidden_dim, nclass)

        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()
        from dgl.nn.pytorch import GCN2Conv
        # from torch_geometric.nn import GATConv,GCN2Conv
        for _ in range(4):
            self.convs.append(Ortho_GCNII(64, 64, 2, 2, 0.1, False, variant=False))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, 64))
        self.fcs.append(nn.Linear(64, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()  # gai

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        if norm == 'none':
            self.layers = nn.ModuleList([SpecLayer(nheads + 1, nclass, prop_dropout, norm=norm) for i in range(nlayer)])
        else:
            self.layers = nn.ModuleList(
                [SpecLayer(nheads + 1, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)])
    def forward(self, graph,  diseases, mirnas,  training=True):



        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        feats = self.G.ndata.pop('z')
        # self.G.apply_nodes(lambda nodes: {'x': self.dropout1(self.d_fc(nodes.data['d_sim1']))}, self.disease_nodes)
        # self.G.apply_nodes(lambda nodes: {'x': self.dropout1(self.m_fc(nodes.data['m_sim1']))}, self.mirna_nodes)
        # feats = self.G.ndata.pop('z')



        #???????????

        if training:  # Training Mode
            X = feats
            x = X
            u = self.u
            e = self.e
            S = self.S
            adj = self.adj
            src_nodes, dst_nodes = graph.edges()
            edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            edge_index = edge_index.to(torch.int64)
            feat0 = []
            N = e.size(0)
            ut = u.permute(1, 0)

            if self.norm == 'none':
                h = self.feat_dp1(x)
                h = self.feat_encoder(h)
                h = self.feat_dp2(h)
            else:
                h = self.feat_dp1(x)
                h = self.linear_encoder(h)

            eig = self.eig_encoder(e) # [N, d]

            _layers = []
            _layers.append(eig)
            for i, con in enumerate(self.convs):
                eig = F.dropout(eig, 0.1, training=self.training)
                eig = self.act_fn(con(eig, adj, _layers[0], 0.5, 0.1, i + 1))
            # eig = F.dropout(eig, 0.5, training=self.training)









            mha_eig = self.mha_norm(eig)
            mha_eig = drop_node(mha_eig, 0.3, True)
            # mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
            # eig = eig + self.mha_dropout(mha_eig)
            from torch_geometric.nn import GATConv,GATv2Conv,GCN2Conv
            gatv2conv = GATv2Conv(64,128,heads=4)
            res = gatv2conv(mha_eig,edge_index)

            res = res.flatten(1)
            res = self.baixu(res)
            res = torch.tanh(res)
            res = th.log_softmax(self.mlp(res), dim=-1)


            eig = res


            ffn_eig = self.ffn_norm(eig)
            ffn_eig = self.ffn(ffn_eig)
            eig = eig + self.ffn_dropout(ffn_eig)

            new_e = self.decoder(eig)  # [N, m]
            for conv in self.layers:
                basic_feats = [h]
                utx = ut @ h
                for i in range(self.nheads):
                    basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]
                basic_feats = torch.stack(basic_feats, axis=1)  # [N, m, d]
                h = conv(basic_feats)

            if self.norm == 'none':
                 h = h
            else:
                h = self.feat_dp2(h)
                #》
                h = self.classify(h)
                #?
            X = h


            feat0 = X
            h_d = feat0[:self.num_diseases]

            h_m = feat0[self.num_diseases:]
            #
            #
            h_m = self.dropout1(F.elu(self.m_fc1(h_m)))     # (495,64)
            h_d = self.dropout1(F.elu(self.d_fc1(h_d)))     # （383,64）
            # (878,64)
            h = th.cat((h_d, h_m), dim=0)

            # 这里的disease和mirnas就是顶点，其对应位置就顶点之间存在边的label：0或者1
            # 疾病顶点特征
            h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
            # mirnas顶点的特征
            h_mirnas = h[mirnas]
            #
            h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)
            predict_score = th.sigmoid(self.predict(h_concat))
            return predict_score
        else:
            X = feats
            x = X
            u = self.u
            e = self.e
            S = self.S
            adj = self.adjmax
            src_nodes, dst_nodes = graph.edges()
            edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            edge_index = edge_index.to(torch.int64)
            feat0 = []
            N = e.size(0)
            ut = u.permute(1, 0)

            if self.norm == 'none':
                h = self.feat_dp1(x)
                h = self.feat_encoder(h)
                h = self.feat_dp2(h)
            else:
                h = self.feat_dp1(x)
                h = self.linear_encoder(h)

            eig = self.eig_encoder(e)  # [N, d]

            _layers = []
            _layers.append(eig)
            for i, con in enumerate(self.convs):
                eig = F.dropout(eig, 0.1, training=self.training)
                eig = self.act_fn(con(eig, adj, _layers[0], 0.5, 0.1, i + 1))
            # eig = F.dropout(eig, 0.5, training=self.training)

            mha_eig = self.mha_norm(eig)
            mha_eig = drop_node(mha_eig, 0.3, True)
            # mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
            # eig = eig + self.mha_dropout(mha_eig)
            from torch_geometric.nn import GATConv, GATv2Conv, GCN2Conv
            gatv2conv = GATv2Conv(64, 128, heads=4)
            res = gatv2conv(mha_eig, edge_index)

            res = res.flatten(1)
            res = self.baixu(res)
            res = torch.tanh(res)
            res = th.log_softmax(self.mlp(res), dim=-1)

            eig = res

            ffn_eig = self.ffn_norm(eig)
            ffn_eig = self.ffn(ffn_eig)
            eig = eig + self.ffn_dropout(ffn_eig)

            new_e = self.decoder(eig)  # [N, m]
            for conv in self.layers:
                basic_feats = [h]
                utx = ut @ h
                for i in range(self.nheads):
                    basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]
                basic_feats = torch.stack(basic_feats, axis=1)  # [N, m, d]
                h = conv(basic_feats)

            if self.norm == 'none':
                h = h
            else:
                h = self.feat_dp2(h)
                # 》
                h = self.classify(h)
                # ?
            X = h

            feat0 = X
            h_d = feat0[:self.num_diseases]

            h_m = feat0[self.num_diseases:]
            #
            #
            h_m = self.dropout1(F.elu(self.m_fc1(h_m)))  # (495,64)
            h_d = self.dropout1(F.elu(self.d_fc1(h_d)))  # （383,64）
            # (878,64)
            h = th.cat((h_d, h_m), dim=0)

            # 这里的disease和mirnas就是顶点，其对应位置就顶点之间存在边的label：0或者1
            # 疾病顶点特征
            h_diseases = h[diseases]  # disease中有重复的疾病名称;(17376,64)
            # mirnas顶点的特征
            h_mirnas = h[mirnas]
            #
            h_concat = th.cat((h_diseases, h_mirnas), 1)  # (17376,128)
            predict_score = th.sigmoid(self.predict(h_concat))
            return predict_score





def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):

        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x





