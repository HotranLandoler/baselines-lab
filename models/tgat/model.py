import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_sparse import SparseTensor

from models.graph_smote import GraphSmote
from models.tgat.layers import TimeEncode


class TGAT(torch.nn.Module):
    """https://github.com/hxttkl/DGraph_Experiments"""
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, edge_dim=32):
        super().__init__()
        self.attention_act = torch.nn.functional.tanh

        self.time_enc = TimeEncode(32)

        self.lin = torch.nn.Linear(in_channels, hid_channels)

        self.conv = TransformerConv(hid_channels, hid_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

        self.out = torch.nn.Linear(hid_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor | SparseTensor, data: Data,
                encode_degree=True, encode_interval=True, **kwargs):
        rel_t = data.node_time[data.edge_index[0]].view(-1, 1) - data.edge_time
        rel_t_enc = self.time_enc(rel_t.to(data.x.dtype))

        h1 = self.lin(x)
        h1 = F.relu(h1)

        # h1, label_scores = self.conv(h1, data.edge_index, rel_t_enc)
        h1 = self.conv(h1, data.edge_index, rel_t_enc)

        # Output classification result
        out = self.out(h1)
        out = F.log_softmax(out, dim=1)

        return h1, out

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.conv.reset_parameters()
        self.out.reset_parameters()
