import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_sparse import SparseTensor

from models.graph_smote import GraphSmote
from models.tgat.layers import (TimeEncode, DegreeEncoder, TemporalFrequencyEncoder, MlpDropTransformerConv,
                                MutualAttention, MutualAttentionSingleFactor, UnifiedAttention)


class TGAT(torch.nn.Module):
    """https://github.com/hxttkl/DGraph_Experiments"""
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, edge_dim=32,
                 drop=False, encode_as_embedding=False):
        super().__init__()
        encoding_dim = 8
        self.attention_act = torch.nn.functional.tanh
        self.encode_as_embedding = encode_as_embedding

        self.time_enc = TimeEncode(32)
        self.degree_enc = DegreeEncoder(encoding_dim)
        self.temporal_frequency_enc = TemporalFrequencyEncoder(encoding_dim)

        self.mutual_attn = MutualAttentionSingleFactor()
        # self.mutual_attn = MutualAttention(encoding_dim)
        self.unified_attn = UnifiedAttention(encoding_dim)

        self.w_enc = torch.nn.Linear(encoding_dim, encoding_dim)
        self.w_x = torch.nn.Linear(hid_channels, encoding_dim)

        self.lin = torch.nn.Linear(in_channels, hid_channels)

        if drop:
            self.conv = MlpDropTransformerConv(hid_channels, hid_channels // 2,
                                               hidden_channels=8,
                                               heads=2,
                                               dropout=0.1, edge_dim=edge_dim)
        else:
            self.conv = TransformerConv(hid_channels, hid_channels // 2, heads=2,
                                        dropout=0.1, edge_dim=edge_dim)

        # self.conv1 = TransformerConv(32, 32 // 2, heads=2,
        #                              dropout=0.1, edge_dim=edge_dim)
        self.out = torch.nn.Linear(hid_channels, out_channels)

        self.lin_combine = torch.nn.Linear(hid_channels + encoding_dim, hid_channels)

    def forward(self, x: Tensor, edge_index: Tensor | SparseTensor, data: Data,
                encode_degree=True, encode_interval=True, **kwargs):
        rel_t = data.node_time[data.edge_index[0]].view(-1, 1) - data.edge_time
        rel_t_enc = self.time_enc(rel_t.to(data.x.dtype))

        h1 = self.lin(x)
        h1 = F.relu(h1)

        # h1, label_scores = self.conv(h1, data.edge_index, rel_t_enc)
        h1 = self.conv(h1, data.edge_index, rel_t_enc)

        if encode_interval:
            temporal_frequency_enc = self.temporal_frequency_enc(
                data.node_mean_out_time_interval)

        if encode_degree:
            degree_enc = self.degree_enc(data.node_out_degree)

        # temporal_frequency_enc, degree_enc = self.mutual_attn(temporal_frequency_enc, degree_enc)
        # temporal_frequency_enc, degree_enc = self.unified_attn(temporal_frequency_enc, degree_enc)
        # diff = temporal_frequency_enc - degree_enc

        # context = self.lin_interval(temporal_frequency_enc) + self.lin_degree(degree_enc)

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

        y_new = None
        train_mask_new = None
        if self.training:
            h1, y_new, train_mask_new = GraphSmote.recon_upsample(h1, data.y, data.train_mask)

        if self.encode_as_embedding:
            # Output node embedding
            out = h1
        else:
            # Output classification result
            out = self.out(h1)
            out = F.log_softmax(out, dim=1)

        return out, y_new, train_mask_new

    def reset_parameters(self):
        # self.time_enc.reset_parameters()
        self.lin.reset_parameters()
        self.conv.reset_parameters()
        # self.conv1.reset_parameters()
        self.out.reset_parameters()

        self.degree_enc.reset_parameters()
        self.temporal_frequency_enc.reset_parameters()

        self.lin_combine.reset_parameters()
