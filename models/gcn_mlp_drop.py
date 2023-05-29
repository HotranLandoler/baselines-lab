import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, Sigmoid

from models.gcn_drop import DropGCN, GNNLayer, BbGCN


class MlpDropGCN(DropGCN):
    """Perform *DropMessage* on GCN with **rate** predicted from an MLP model"""
    def __init__(self, feature_num: int, output_num: int):
        super().__init__(feature_num=feature_num,
                         output_num=output_num)
        self.gnn1 = AdaptiveGNNLayer(feature_num, 64)
        self.gnn2 = AdaptiveGNNLayer(64, output_num)


class AdaptiveGNNLayer(GNNLayer):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 transform_first: bool = False):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         heads=heads,
                         transform_first=transform_first)
        self.backbone = AdaptiveBbGCN(in_channels=in_channels)


class AdaptiveBbGCN(BbGCN):
    def __init__(self, in_channels: int, hidden_channels=8):
        """Init GCN with MLP-DropMessage

        Args:
            in_channels: input dimension of MLP.
            hidden_channels: hidden dimension of MLP.
        """
        super().__init__()
        self.pre_transform_linear = Linear(in_channels, hidden_channels)
        self.mlp = Sequential(Linear(3 * hidden_channels, 1),
                              Sigmoid())

    def message(self, x_i: Tensor, x_j: Tensor, drop_rate: float):
        if not self.training:
            return x_j

        x_i_transformed = self.pre_transform_linear(x_i)
        x_j_transformed = self.pre_transform_linear(x_j)
        diff = x_i_transformed - x_j_transformed
        x_cat = torch.cat([x_i_transformed, x_j_transformed, diff], dim=1)
        drop_rate_mlp = self.mlp(x_cat)
        # print(f"MLP output: {drop_rate_mlp}")
        print("Dropping...")

        # drop messages
        for i, message in enumerate(x_j):
            F.dropout(x_j[i], p=drop_rate_mlp[i].item())

        # print(f"After drop: {x_j}")
        print("Dropped.")

        return x_j
