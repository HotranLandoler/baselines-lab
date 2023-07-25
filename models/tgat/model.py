import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_sparse import SparseTensor

from models.tgat.layers import TimeEncode, MlpDropTransformerConv


class TGAT(torch.nn.Module):
    """https://github.com/hxttkl/DGraph_Experiments"""
    def __init__(self, in_channels: int, out_channels: int, edge_dim=32):
        super().__init__()
        self.time_enc = TimeEncode(32)
        self.lin = torch.nn.Linear(in_channels, 32)
        # self.conv = TransformerConv(32, 32 // 1, heads=1,
        #                             dropout=0.1, edge_dim=edge_dim)
        self.conv = MlpDropTransformerConv(32, 32 // 2,
                                           hidden_channels=8,
                                           heads=2,
                                           dropout=0.1, edge_dim=edge_dim)
        self.out = torch.nn.Linear(32, out_channels)

        self.lin_degree = torch.nn.Linear(1, 8)
        self.lin_combine = torch.nn.Linear(8 + 32, 32)

    def forward(self, x: Tensor, edge_index: Tensor | SparseTensor, data: Data,
                encode_degree=True):
        rel_t = data.node_time[data.edge_index[0]].view(-1, 1) - data.edge_time
        rel_t_enc = self.time_enc(rel_t.to(data.x.dtype))

        h1 = self.lin(data.x)
        h1 = F.relu(h1)

        if encode_degree:
            degree_enc = self.lin_degree(data.node_out_degree)
            degree_enc = F.relu(degree_enc)

            h1 = self.lin_combine(torch.concat((h1, degree_enc), dim=1))

        h1 = self.conv(h1, data.edge_index, rel_t_enc)

        # H2GCN D1
        # h1 = torch.concat((data.x, h1), dim=1)
        out = self.out(h1)
        return F.log_softmax(out, dim=1)

    def reset_parameters(self):
        # self.time_enc.reset_parameters()
        self.lin.reset_parameters()
        self.conv.reset_parameters()
        # self.conv1.reset_parameters()
        self.out.reset_parameters()

        self.lin_degree.reset_parameters()
        self.lin_combine.reset_parameters()
