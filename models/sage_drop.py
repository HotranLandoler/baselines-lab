import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Sequential, Linear
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import SAGEConv

import models.gfca.layers as cafd


class DropSAGE(torch.nn.Module):
    def __init__(self, feature_num, hidden_num, output_num):
        super().__init__()
        encoding_dim = 8
        self.gnn1 = SAGEConv(feature_num, hidden_num)
        self.gnn2 = DropSAGEConv(hidden_num, hidden_num)
        self.attention_act = torch.nn.functional.tanh
        self.degree_enc = cafd.DegreeEncoder(encoding_dim)
        self.temporal_frequency_enc = cafd.TemporalFrequencyEncoder(encoding_dim)

        self.w_enc = torch.nn.Linear(encoding_dim, encoding_dim)
        self.w_x = torch.nn.Linear(hidden_num, encoding_dim)

        self.lin_combine = torch.nn.Linear(hidden_num + encoding_dim, hidden_num)
        self.out = torch.nn.Linear(hidden_num, output_num)

    def forward(self, x: Tensor, edge_index: Adj, data: Data, **kwargs):
        x = self.gnn1(x, edge_index)
        h1 = F.relu(x)
        # h1 = F.dropout(x, p=0.1, training=self.training)
        h1 = self.gnn2(h1, edge_index)

        temporal_frequency_enc = self.temporal_frequency_enc(
            data.node_mean_out_time_interval)

        degree_enc = self.degree_enc(data.node_out_degree)

        encodings = torch.stack((temporal_frequency_enc, degree_enc), dim=1)
        encodings_proj = self.attention_act(self.w_enc(encodings))
        x_proj = self.attention_act(self.w_x(h1)).unsqueeze(-1)

        score = torch.bmm(encodings_proj, x_proj)
        score = torch.nn.functional.softmax(score, dim=1)

        context = (encodings[:, 0, :] * score[:, 0] +
                   encodings[:, 1, :] * score[:, 1])

        h1 = self.lin_combine(torch.concat((h1, context), dim=1))

        out = self.out(h1)
        out = F.log_softmax(out, dim=1)

        return h1, out

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()
        self.out.reset_parameters()

        self.degree_enc.reset_parameters()
        self.temporal_frequency_enc.reset_parameters()

        self.lin_combine.reset_parameters()


class DropSAGEConv(SAGEConv):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__(in_channels,
                         out_channels)
        self.drop_rate = None
        self.pre_transform_linear = Linear(out_channels, 8)
        self.label_score_predictor = torch.nn.Linear(8, 1)

    def message(self,
                x_i: Tensor,
                x_j: Tensor,
                index: Tensor,
                ptr: Tensor | None,
                size_i: int | None):
        out = super().message(x_j)
        if not self.training:
            # return out * (1.0 - self.drop_rate)
            return out

        x_i_transformed = self.pre_transform_linear(x_i)
        x_j_transformed = self.pre_transform_linear(x_j)
        drop_rate = self._get_drop_rate_by_label_aware_distance(x_i_transformed, x_j_transformed)

        self.drop_rate = drop_rate.detach()

        # print(f"drop rate[0]: {self.drop_rate[0][0].item()}")
        # print(f"drop rate[1]: {self.drop_rate[1][0].item()}")
        # print(f"drop rate[2]: {self.drop_rate[2][0].item()}")

        out = cafd.multi_dropout(out, probability=self.drop_rate)

        return out

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'pre_transform_linear'):
            self.pre_transform_linear.reset_parameters()
        if hasattr(self, 'label_score_predictor'):
            self.label_score_predictor.reset_parameters()

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
