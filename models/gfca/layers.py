from typing import Callable

import math

import torch
import numpy as np
import torch_geometric.utils as pyg_utils
from torch import Tensor
from torch.nn import Linear, Sequential, LayerNorm, GELU, Dropout
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import OptTensor, PairTensor, Adj
from torch_sparse import SparseTensor


class TemporalFrequencyEncoder(torch.nn.Module):
    def __init__(self, expand_dim: int):
        super().__init__()
        self.linear = Linear(1, expand_dim)

    def forward(self, temporal_frequencies: Tensor) -> Tensor:
        temporal_frequencies_enc = self.linear(temporal_frequencies)
        temporal_frequencies_enc = torch.nn.functional.relu(temporal_frequencies_enc)
        return temporal_frequencies_enc

    def reset_parameters(self):
        self.linear.reset_parameters()


class DegreeEncoder(torch.nn.Module):
    def __init__(self, expand_dim: int):
        super().__init__()
        self.linear = Linear(1, expand_dim)

    def forward(self, degrees: Tensor) -> Tensor:
        # degrees = self._degree_kernel(degrees)
        degrees_enc = self.linear(degrees)
        degrees_enc = torch.nn.functional.relu(degrees_enc)
        return degrees_enc

    def reset_parameters(self):
        self.linear.reset_parameters()

    @staticmethod
    def _degree_kernel(degrees: Tensor, a=0.5, b=0.0) -> Tensor:
        return 1.0 / (1.0 + (torch.exp(-a * degrees) + b))


class MlpDropTransformerConv(TransformerConv):
    drop_rate: Tensor | None
    pre_transform_linear: Linear
    mlp: Linear

    """Drop message based on mlp"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 # label_predictor: Callable[[], Tensor],
                 heads: int = 1,
                 dropout: float = 0.,
                 edge_dim: int | None = None):
        super().__init__(in_channels,
                         out_channels,
                         heads=heads,
                         dropout=dropout,
                         edge_dim=edge_dim)
        self.drop_rate = None
        self.pre_transform_linear = Linear(out_channels, hidden_channels)
        self.mlp = Linear(3 * hidden_channels, 1)
        # self.label_predictor = Linear(hidden_channels, 1)
        self.label_score_predictor = torch.nn.Linear(8, 1)
        # self.mlp2 = Linear(hidden_channels, 1)
        # self.sb = Sequential(Linear(3 * hidden_channels, hidden_channels),
        #                      GELU(),
        #                      Dropout(p=0.1),
        #                      # LayerNorm(hidden_channels),
        #                      Linear(hidden_channels, 1))

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: int | None,
                # label_score_i: Tensor, label_score_j: Tensor
                ) -> Tensor:
        out = super().message(query_i, key_j, value_j, edge_attr, index, ptr, size_i)
        if not self.training:
            # return out * (1.0 - self.drop_rate)
            return out

        # if self.drop_rate is None:
        x_i_transformed = self.pre_transform_linear(query_i)
        # x_i_transformed = torch.nn.functional.tanh(x_i_transformed)
        x_j_transformed = self.pre_transform_linear(key_j)
        # x_j_transformed = torch.nn.functional.tanh(x_j_transformed)

        # drop_rate = self._get_drop_rate_by_cosine_similarity(x_i_transformed, x_j_transformed)
        # drop_rate = self._get_drop_rate_by_mlp(x_i_transformed, x_j_transformed,
        #                                        index, ptr, size_i)
        drop_rate = self._get_drop_rate_by_label_aware_distance(x_i_transformed, x_j_transformed)
        # drop_rate = drop_rate.repeat(1, 2).unsqueeze(-1)

        self.drop_rate = drop_rate.detach()

        # print(f"drop rate[0]: {self.drop_rate[0][0].item()}")
        # print(f"drop rate[1]: {self.drop_rate[1][0].item()}")
        # print(f"drop rate[2]: {self.drop_rate[2][0].item()}")

        out = _multi_dropout(out, probability=self.drop_rate)
        # out = _drop_edge_by_rate(out, drop_rate=self.drop_rate)

        # out = _drop_edge(out, cos_similarity)

        return out

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'pre_transform_linear'):
            self.pre_transform_linear.reset_parameters()
        if hasattr(self, 'mlp'):
            self.mlp.reset_parameters()
        if hasattr(self, 'label_score_predictor'):
            self.label_score_predictor.reset_parameters()

    def _get_drop_rate_by_cosine_similarity(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        cos_similarity = torch.cosine_similarity(
            x_i,
            x_j,
            dim=-1).reshape(-1, self.heads, 1)
        return torch.nn.functional.relu(-cos_similarity)

    def _get_drop_rate_by_mlp(self, x_i: Tensor, x_j: Tensor,
                              index: Tensor, ptr: OptTensor,
                              size_i: int | None) -> Tensor:
        diff = x_i - x_j
        # diff = (x_i * x_j).sum(-1, keepdims=True)
        # diff = self.pre_transform_linear(query_i * key_j)
        x_cat = torch.cat([x_i, x_j, diff], dim=-1)
        drop_rate = self.mlp(x_cat)
        # drop_rate = torch.nn.functional.sigmoid(drop_rate)
        # drop_rate = torch.nn.functional.relu(drop_rate)
        # drop_rate = torch.exp(-drop_rate)
        drop_rate = pyg_utils.softmax(drop_rate, index, ptr, size_i)
        return drop_rate

    def _get_drop_rate_by_label_aware_distance(self, label_score_i: Tensor, label_score_j: Tensor) -> (Tensor, Tensor):
        """The Distance Function in PC-GNN"""
        x_i = self.label_score_predictor(label_score_i)
        x_j = self.label_score_predictor(label_score_j)
        label_i = torch.nn.functional.sigmoid(x_i)
        label_j = torch.nn.functional.sigmoid(x_j)
        drop_rate = torch.abs(label_i - label_j)
        # drop_rate = torch.abs(label_score_i - label_score_j)

        # return drop_rate[:, 0].reshape(-1, 1)
        return drop_rate


def _drop_edge(message: Tensor, similarity: Tensor, threshold=0.1) -> Tensor:
    mask: Tensor = similarity >= threshold
    return mask * message


def _drop_edge_by_rate(message: Tensor, drop_rate: Tensor) -> Tensor:
    mask: Tensor = drop_rate <= 0.
    return mask * message


def _multi_dropout(x: Tensor, probability: Tensor) -> Tensor:
    mask: Tensor = torch.rand_like(x) > probability
    return mask * x  # / (1.0 - probability)
