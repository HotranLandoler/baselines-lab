import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU


class MLP(torch.nn.Module):
    mlp: torch.nn.Module

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(Linear(in_channels, hidden_channels),
                                       ReLU(),
                                       Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index):
        x = self.mlp(x)
        return x.log_softmax(dim=-1)
