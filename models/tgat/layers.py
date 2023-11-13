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
        degrees = self._degree_kernel(degrees)
        degrees_enc = self.linear(degrees)
        degrees_enc = torch.nn.functional.relu(degrees_enc)
        return degrees_enc

    def reset_parameters(self):
        self.linear.reset_parameters()

    @staticmethod
    def _degree_kernel(degrees: Tensor, a=0.5, b=0.0) -> Tensor:
        return 1.0 / (1.0 + (torch.exp(-a * degrees) + b))


class MutualAttention(torch.nn.Module):
    def __init__(self, encoding_dim: int):
        super().__init__()
        self.attn_weight = torch.nn.Parameter(torch.empty(encoding_dim, encoding_dim))
        self.reset_parameters()

    def forward(self, p: Tensor, q: Tensor) -> (Tensor, Tensor):
        p_a_q = torch.bmm((p @ self.attn_weight).unsqueeze(-1), q.unsqueeze(1))
        p_a_q = torch.nn.functional.tanh(p_a_q)
        attn_p = torch.nn.functional.softmax(p_a_q.mean(-1), dim=-1)
        attn_q = torch.nn.functional.softmax(p_a_q.mean(1), dim=-1)
        return p * attn_p, q * attn_q

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.attn_weight, a=math.sqrt(5))


class MutualAttentionSingleFactor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_weight = torch.nn.Parameter(torch.randn(size=[1]))
        self.reset_parameters()

    def forward(self, p: Tensor, q: Tensor) -> (Tensor, Tensor):
        p_a_q = torch.bmm((p.unsqueeze(-1) @ self.attn_weight).unsqueeze(-1), q.unsqueeze(1))
        p_a_q = torch.nn.functional.tanh(p_a_q)
        attn_p = torch.nn.functional.softmax(p_a_q.mean(-1), dim=-1)
        attn_q = torch.nn.functional.softmax(p_a_q.mean(1), dim=-1)
        # print(f"Weight: {self.attn_weight.item()}")
        return p * attn_p, q * attn_q

    def reset_parameters(self):
        pass


class UnifiedAttention(torch.nn.Module):
    def __init__(self, encoding_dim: int):
        super().__init__()
        self.encoding_dim = encoding_dim

        self.linear_q = Linear(2 * encoding_dim, 2 * encoding_dim)
        self.linear_k = Linear(2 * encoding_dim, 2 * encoding_dim)
        self.linear_v = Linear(2 * encoding_dim, 2 * encoding_dim)

        self.linear_qk = Linear(2 * encoding_dim, 2 * encoding_dim)
        self.linear_m = Linear(2 * encoding_dim, 2)

        self.linear_out = Linear(2 * encoding_dim, 2 * encoding_dim)
        self.reset_parameters()

    def forward(self, x: Tensor, y: Tensor) -> (Tensor, Tensor):
        z = torch.cat((x, y), dim=1)
        q = self.linear_q(z)
        k = self.linear_k(z)
        v = self.linear_v(z)
        gsa = v @ self._bgdp(q, k) + z
        z = self.linear_out(gsa)
        return (z[:, :self.encoding_dim],
                z[:, self.encoding_dim:])

    def _bgdp(self, q: Tensor, k: Tensor) -> Tensor:
        # m = self.linear_m(self.linear_qk(q) * self.linear_qk(k))
        # m = torch.nn.functional.sigmoid(m)
        # out = (q * m[:, [0]]).T @ (k * m[:, [1]]) / math.sqrt(self.encoding_dim)
        out = q.T @ k / math.sqrt(self.encoding_dim)
        return torch.nn.functional.softmax(out)

    def reset_parameters(self):
        pass


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
        self.drop_rate = None
        self.pre_transform_linear = Linear(out_channels, hidden_channels)
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

        # if self.drop_rate is None:
        x_i_transformed = self.pre_transform_linear(query_i)
        # x_i_transformed = torch.nn.functional.tanh(x_i_transformed)
        x_j_transformed = self.pre_transform_linear(key_j)
        # x_j_transformed = torch.nn.functional.tanh(x_j_transformed)

        # diff = x_i_transformed - x_j_transformed
        # diff = (x_i_transformed * x_j_transformed).sum(-1, keepdims=True)
        # diff = self.pre_transform_linear(query_i * key_j)
        # x_cat = torch.cat([x_i_transformed, x_j_transformed, diff], dim=-1)
        cos_similarity = torch.cosine_similarity(
            x_i_transformed,
            x_j_transformed,
            dim=-1).reshape(-1, self.heads, 1)
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
        self.drop_rate = drop_rate.detach()

        print(f"drop rate[0]: {self.drop_rate[0][0].item()}")

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


def _drop_edge(message: Tensor, similarity: Tensor, threshold=0.1) -> Tensor:
    mask: Tensor = similarity >= threshold
    return mask * message


def _drop_edge_by_rate(message: Tensor, drop_rate: Tensor) -> Tensor:
    mask: Tensor = drop_rate <= 0.
    return mask * message


def _multi_dropout(x: Tensor, probability: Tensor) -> Tensor:
    mask: Tensor = torch.rand_like(x) > probability
    return mask * x  # / (1.0 - probability)
