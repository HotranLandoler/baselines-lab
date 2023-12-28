import torch.nn
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch_sparse import SparseTensor


class GraphSAGE(torch.nn.Module):
    """Wrapper for `torch_geometric.nn.models.GraphSAGE`"""
    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 dropout: float):
        super().__init__()

        self.sage = pyg_nn.GraphSAGE(in_channels=in_channels,
                                     hidden_channels=hidden_channels,
                                     out_channels=out_channels,
                                     num_layers=num_layers,
                                     dropout=dropout)

    def reset_parameters(self):
        self.sage.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor | SparseTensor, **kwargs):
        output = self.sage(x, edge_index)

        return output, output.log_softmax(dim=-1)
