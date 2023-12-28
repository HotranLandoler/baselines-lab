import torch
import torch_geometric.nn
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU


class MLP(torch.nn.Module):
    mlp: torch.nn.Module

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index, **kwargs):
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.lin2(x)
        return x, x.log_softmax(dim=-1)
