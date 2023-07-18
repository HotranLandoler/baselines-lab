from typing import Union

import torch.nn
import torch.nn.functional as f
from torch import Tensor
from torch_geometric.nn.conv import GCNConv
from torch_sparse import SparseTensor


class GCN(torch.nn.Module):
    """
    GCN Model from
    https://github.com/DGraphXinye/DGraphFin_baseline/blob/master/models/gcn.py
    """
    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 dropout: float,
                 batch_norm=False):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.batchnorm = batch_norm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor], **kwargs):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = f.relu(x)
            x = f.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)
