import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_sparse
import torch_geometric.utils as pyg_utils
from torch import FloatTensor, Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import models.h2gcn.utils as h2_utils


class H2GCN(torch.nn.Module):
    def __init__(self, data: Data, num_features, num_hidden, num_classes, dropout, layer_num=2, device='cpu'):
        super(H2GCN, self).__init__()

        self.linear1 = torch.nn.Linear(num_features, num_hidden)

        self.linear2 = torch.nn.Linear(num_hidden + 2 * num_hidden + 4 * num_hidden, num_classes)

        self.dropout = dropout
        self.layer_num = layer_num
        self.data = data
        self.device = device

        temp_loop_edge_index, _ = pyg_utils.add_self_loops(self.data.edge_index)
        sparse_adj_tensor = h2_utils.edge_index_to_sparse_tensor_adj(temp_loop_edge_index)

        k_hop_adjs = []
        k_hop_edge_index = []
        k_hop_adjs.append(sparse_adj_tensor)

        for i in range(self.layer_num-1):
            temp_adj_adj = torch.sparse.mm(k_hop_adjs[i], sparse_adj_tensor)

            k_hop_adjs.append(temp_adj_adj)
            k_hop_edge_index.append(temp_adj_adj._indices())

        self.k_hop_edge_index = k_hop_edge_index

        for i in range(self.layer_num-1):
            self.k_hop_edge_index[i], _ = pyg_utils.remove_self_loops(self.k_hop_edge_index[i])
            if i == 0:
                self.k_hop_edge_index[i] = h2_utils.remove_edges(
                    self.k_hop_edge_index[i], self.data.edge_index).to(self.device)
            else:
                self.k_hop_edge_index[i] = h2_utils.remove_edges(
                    self.k_hop_edge_index[i], self.k_hop_edge_index[i-1]).to(self.device)

        self.norm_adjs = []
        self.norm_adjs.append(h2_utils.gcn_norm(self.data.edge_index, self.data.y.shape[0], self.device))
        self.norm_adjs.append(h2_utils.gcn_norm(self.k_hop_edge_index[0], self.data.y.shape[0], self.device))

    def forward(self, x, edge_index, **kwargs):
        h = self.linear1(self.data.x)
        h = F.relu(h)
        final_h = h

        # first layer
        first_hop_h = torch.sparse.mm(self.norm_adjs[0], h)
        second_hop_h = torch.sparse.mm(self.norm_adjs[1], h)
        R1 = torch.cat([first_hop_h, second_hop_h], dim=1)

        # second layer
        first_hop_h2 = torch.sparse.mm(self.norm_adjs[0], R1)
        second_hop_h2 = torch.sparse.mm(self.norm_adjs[1], R1)
        R2 = torch.cat([first_hop_h2, second_hop_h2], dim=1)

        final_h = torch.cat([final_h, R1], dim=1)
        final_h = torch.cat([final_h, R2], dim=1)
        final_h = F.dropout(final_h, p=self.dropout, training=self.training)
        final_h = self.linear2(final_h)

        return F.log_softmax(final_h, 1)

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()


class H2GCN_ORIGINIAL(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            class_dim: int,
            k: int = 2,
            dropout: float = 0.5,
            use_relu: bool = True
    ):
        super(H2GCN_ORIGINIAL, self).__init__()
        self.dropout = dropout
        self.k = k
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, x: FloatTensor, adj: torch.sparse.Tensor, **kwargs) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return torch.softmax(torch.mm(r_final, self.w_classify), dim=1)


class H2GCN_EGO(torch.nn.Module):
    """
    GCN Model from
    https://github.com/DGraphXinye/DGraphFin_baseline/blob/master/models/gcn.py
    """
    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 dropout: float):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True,
                                  add_self_loops=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True,
                                  add_self_loops=True))
        self.linear = torch.nn.Linear(in_features=in_channels + hidden_channels + out_channels,
                                      out_features=out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: SparseTensor, **kwargs):
        intermediate_outputs = [x]
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            intermediate_outputs.append(x)
        x = self.convs[-1](x, edge_index)
        intermediate_outputs.append(x)
        x = torch.cat(intermediate_outputs, dim=1)
        x = self.linear(x)
        return x.log_softmax(dim=-1)
