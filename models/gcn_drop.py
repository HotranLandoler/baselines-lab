import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Sequential
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing


class DropGCN(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super().__init__()
        self.gnn1 = GNNLayer(feature_num, 64)
        self.gnn2 = GNNLayer(64, output_num)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate=0.0):
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
        self.edge_weight = None

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        y = self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate)
        return y

    def message(self, x_j: Tensor, drop_rate: float):
        # normalize
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)

        if not self.training:
            return x_j

        # drop messages
        x_j = F.dropout(x_j, drop_rate)

        return x_j


def _glorot(tensor: Tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def _zeros(tensor: Tensor):
    if tensor is not None:
        tensor.data.fill_(0)
