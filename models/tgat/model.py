import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_sparse import SparseTensor

from models.tgat.layers import TimeEncode, DegreeEncoder, TemporalFrequencyEncoder, MlpDropTransformerConv


class TGAT(torch.nn.Module):
    """https://github.com/hxttkl/DGraph_Experiments"""
    def __init__(self, in_channels: int, out_channels: int, edge_dim=32,
                 drop=True):
        super().__init__()
        hid_channels = 32
        encoding_dim = 8
        self.attention_act = torch.nn.functional.tanh

        self.time_enc = TimeEncode(32)
        self.degree_enc = DegreeEncoder(encoding_dim)
        self.temporal_frequency_enc = TemporalFrequencyEncoder(encoding_dim)
        self.w_enc = torch.nn.Linear(encoding_dim, encoding_dim)
        self.w_x = torch.nn.Linear(hid_channels, encoding_dim)

        self.lin = torch.nn.Linear(in_channels, hid_channels)

        if drop:
            self.conv = MlpDropTransformerConv(32, 32 // 2,
                                               hidden_channels=8,
                                               heads=2,
                                               dropout=0.1, edge_dim=edge_dim)
        else:
            self.conv = TransformerConv(32, 32 // 2, heads=2,
                                        dropout=0.1, edge_dim=edge_dim)

        self.conv1 = TransformerConv(32, 32 // 2, heads=2,
                                     dropout=0.1, edge_dim=edge_dim)
        self.out = torch.nn.Linear(32, out_channels)

        # self.lin_degree = torch.nn.Linear(1, 8)
        # self.lin_degree1 = torch.nn.Linear(8, 4)
        self.lin_combine = torch.nn.Linear(32 + 8, 32)

        # self.lin_interval = torch.nn.Linear(1, 8)
        #
        # self.lin_intermediate_results = torch.nn.Linear(32 * 2, 32)

        self.lin_edge_attr = torch.nn.Linear(32 + 172, 32)

    def forward(self, x: Tensor, edge_index: Tensor | SparseTensor, data: Data,
                encode_degree=True, encode_interval=True, **kwargs):
        rel_t = data.node_time[data.edge_index[0]].view(-1, 1) - data.edge_time
        rel_t_enc = self.time_enc(rel_t.to(data.x.dtype))

        h1 = self.lin(data.x)
        # h1 = F.relu(h1)

        if encode_interval:
            temporal_frequency_enc = self.temporal_frequency_enc(
                data.node_mean_out_time_interval)

            # interval_enc = self.lin_interval(data.node_mean_out_time_interval)
            # interval_enc = F.relu(interval_enc)
            # h1 = self.lin_combine(torch.concat((h1, interval_enc), dim=1))

        if encode_degree:
            degree_enc = self.degree_enc(data.node_out_degree)

            # degree_enc = self.lin_degree(data.node_out_degree)
            # degree_enc = F.relu(degree_enc)
            # h1 = self.lin_combine(torch.concat((h1, degree_enc), dim=1))

        encodings = torch.stack((temporal_frequency_enc, degree_enc), dim=1)
        encodings_proj = self.attention_act(self.w_enc(encodings))
        x_proj = self.attention_act(self.w_x(h1)).unsqueeze(-1)

        score = torch.bmm(encodings_proj, x_proj)
        score = torch.nn.functional.softmax(score, dim=1)

        context = (encodings[:, 0, :] * score[:, 0] +
                   encodings[:, 1, :] * score[:, 1])

        h1 = self.lin_combine(torch.concat((h1, context), dim=1))

        # Edge Attr
        # rel_t_enc = self.lin_edge_attr(torch.cat((rel_t_enc,
        #                                           data.edge_attr.view(-1, 1, 172)), dim=-1))
        # rel_t_enc = torch.cat((rel_t_enc, data.edge_attr.view(-1, 1, 172)), dim=-1)

        h1 = self.conv(h1, data.edge_index, rel_t_enc)

        # Layer 2
        # intermediate_results = [h1]
        # h1 = F.relu(h1)
        # h1 = self.conv1(h1, edge_index, rel_t_enc)
        # intermediate_results.append(h1)
        # h1 = torch.cat(intermediate_results, dim=1)
        # h1 = self.lin_intermediate_results(h1)

        out = self.out(h1)
        return F.log_softmax(out, dim=1)

    def reset_parameters(self):
        # self.time_enc.reset_parameters()
        self.lin.reset_parameters()
        self.conv.reset_parameters()
        self.conv1.reset_parameters()
        self.out.reset_parameters()

        self.degree_enc.reset_parameters()
        self.temporal_frequency_enc.reset_parameters()

        # self.lin_degree.reset_parameters()
        # self.lin_degree1.reset_parameters()
        self.lin_combine.reset_parameters()

        # self.lin_interval.reset_parameters()
