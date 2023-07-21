import math

import torch
import torch.nn.functional as F
import torch_sparse
import torch_geometric
from torch import Tensor
from torch.nn import Parameter, Sequential
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing, GCNConv


class DropGCN(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super().__init__()
        self.gnn1 = GNNLayer(feature_num, 64)
        self.gnn2 = GNNLayer(64, output_num)
        # self.gnn1 = GCNConv(feature_num, 64)
        # self.gnn2 = GCNConv(64, output_num)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate=0.0, **kwargs):
        x = self.gnn1(x, edge_index, drop_rate)
        x = F.relu(x)
        x = self.gnn2(x, edge_index, drop_rate)
        return x.log_softmax(dim=-1)

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


class GNNLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 transform_first: bool = False):
        super(GNNLayer, self).__init__()
        self.transform_first = transform_first
        self.backbone = BbGCN()

        # parameters
        self.weight = Parameter(torch.Tensor(heads * in_channels, heads * out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        _glorot(self.weight)
        _zeros(self.bias)
        self.backbone.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        message_drop = drop_rate

        if self.transform_first:
            x = x.matmul(self.weight)

        out = self.backbone(x, edge_index, message_drop)

        if not self.transform_first:
            out = out.matmul(self.weight)
        if self.bias is not None:
            out += self.bias

        return out


class BbGCN(MessagePassing):
    """GCN with DropMessage"""
    def __init__(self):
        super().__init__()
        self.pt = ModelPretreatment(add_self_loops=True, normalize=True)
        self.edge_weight = None

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        edge_index, self.edge_weight = self.pt.pretreatment(x, edge_index)
        dim_size = self._check_input(edge_index, None)[1]
        y = self.propagate(edge_index=edge_index,
                           size=None,
                           x=x,
                           drop_rate=drop_rate,
                           edge_index_target=edge_index.storage.col(),
                           dim_size=dim_size)
        return y

    def message(self,
                x_j: Tensor,
                drop_rate: float,
                edge_index_target: Tensor,
                dim_size: int):
        # normalize
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)

        if not self.training:
            return x_j

        # if drop_rate == 0.0:
        #     return x_j

        # drop messages
        x_j = F.dropout(x_j, drop_rate)

        return x_j


class ModelPretreatment:
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        super(ModelPretreatment, self).__init__()
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def pretreatment(self, x: Tensor, edge_index: Adj):
        # add self loop
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        # normalize
        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                row, col = edge_index
            elif isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, edge_weight


def _glorot(tensor: Tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def _zeros(tensor: Tensor):
    if tensor is not None:
        tensor.data.fill_(0)
