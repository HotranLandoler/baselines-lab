import math

import torch
import numpy as np
import torch_geometric.utils as pyg_utils
from torch import Tensor
from torch.nn import Linear, Sequential, LayerNorm, GELU, Dropout
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import OptTensor


class TimeEncode(torch.nn.Module):
    # https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
    def __init__(self, expand_dim: int, factor=5, is_fixed=False):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        if is_fixed:
            self.basis_freq.requires_grad = False
            self.phase.requires_grad = False

    # @torch.no_grad()
    def forward(self, ts: Tensor) -> Tensor:
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)


class MlpDropTransformerConv(TransformerConv):
    drop_rate: Tensor | None
    pre_transform_linear: Linear
    mlp: Linear

    """Drop message based on mlp"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 heads: int = 1,
                 dropout: float = 0.,
                 edge_dim: int | None = None):
        super().__init__(in_channels,
                         out_channels,
                         heads=heads,
                         dropout=dropout,
                         edge_dim=edge_dim)
        self.pre_transform_linear = Linear(out_channels, hidden_channels)
        self.pre_transform_linear2 = Linear(out_channels, hidden_channels)
        self.mlp = Linear(3 * hidden_channels, hidden_channels)
        self.mlp2 = Linear(hidden_channels, 1)
        # self.sb = Sequential(Linear(3 * hidden_channels, hidden_channels),
        #                      GELU(),
        #                      Dropout(p=0.1),
        #                      # LayerNorm(hidden_channels),
        #                      Linear(hidden_channels, 1))

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: int | None) -> Tensor:
        out = super().message(query_i, key_j, value_j, edge_attr, index, ptr, size_i)
        if not self.training:
            return out * (1.0 - self.drop_rate)

        x_i_transformed = self.pre_transform_linear(query_i)
        x_i_transformed = torch.nn.functional.tanh(x_i_transformed)
        x_j_transformed = self.pre_transform_linear2(key_j)
        x_j_transformed = torch.nn.functional.tanh(x_j_transformed)
        # diff = x_i_transformed - x_j_transformed
        # diff = (x_i_transformed * x_j_transformed).sum(-1, keepdims=True)
        # diff = self.pre_transform_linear(query_i * key_j)
        # x_cat = torch.cat([x_i_transformed, x_j_transformed, diff], dim=-1)
        cos_similarity = torch.cosine_similarity(
            x_i_transformed,
            x_j_transformed,
            dim=-1).view(-1, self.heads, 1)
        drop_rate = torch.nn.functional.relu(-cos_similarity)

        # drop_rate = self.mlp2(cos_similarity)
        # drop_rate = self.mlp(cos_similarity)
        # drop_rate = torch.nn.functional.relu(drop_rate)
        # drop_rate = self.mlp2(drop_rate)

        # drop_rate = self.sb(x_cat)

        # drop_rate = torch.nn.functional.sigmoid(drop_rate)

        # drop_rate = torch.nn.functional.relu(drop_rate)
        # drop_rate = torch.exp(-drop_rate)

        # drop_rate = pyg_utils.softmax(drop_rate, index, ptr, size_i)

        print(f"drop rate[0]: {drop_rate[0][0].item()}")

        out = _multi_dropout(out, probability=drop_rate)
        self.drop_rate = drop_rate

        # cos_similarity = torch.cosine_similarity(
        #     query_i,
        #     key_j,
        #     dim=-1).view(-1, self.heads, 1)
        # out = _drop_edge(out, cos_similarity)

        return out

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'pre_transform_linear'):
            self.pre_transform_linear.reset_parameters()
        if hasattr(self, 'mlp'):
            self.mlp.reset_parameters()


def _drop_edge(message: Tensor, similarity: Tensor, threshold=0.1) -> Tensor:
    mask: Tensor = similarity >= threshold
    return mask * message


def _multi_dropout(x: Tensor, probability: Tensor) -> Tensor:
    mask: Tensor = torch.rand_like(x) > probability
    return mask * x  # / (1.0 - probability)
