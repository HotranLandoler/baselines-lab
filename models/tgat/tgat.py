import torch
import torch.nn.functional as F
import numpy as np
from cogdl.data import Graph
from cogdl.models import BaseModel
from torch_geometric.nn import TransformerConv

from .layers import TimeEncode
from models.model import Model


class TGAT(Model):
    """https://github.com/hxttkl/DGraph_Experiments"""
    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_channels=args.num_features,
            out_channels=args.num_classes
        )

    def __init__(self, in_channels, out_channels, edge_dim=32):
        super().__init__()
        self.time_enc = TimeEncode(32)
        self.lin = torch.nn.Linear(in_channels, 32)
        self.conv = TransformerConv(32, 32 // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)
        self.conv1 = TransformerConv(32, 32 // 2, heads=2,
                                     dropout=0.1, edge_dim=edge_dim)
        self.out = torch.nn.Linear(32, out_channels)

    def forward(self, data: Graph):
        x = data.x
        edge_index = data.edge_index
        t = data.edge_time
        print((data.node_time.device, t.device))

        rel_t = data.node_time[edge_index[0]].view(-1, 1) - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        h1 = self.lin(x)
        h1 = F.relu(h1)
        h1 = self.conv(h1, edge_index, rel_t_enc)
        out = self.out(h1)
        return F.log_softmax(out, dim=1)

    def reset_parameters(self):
        # self.time_enc.reset_parameters()
        self.lin.reset_parameters()
        self.conv.reset_parameters()
        self.conv1.reset_parameters()
        self.out.reset_parameters()
