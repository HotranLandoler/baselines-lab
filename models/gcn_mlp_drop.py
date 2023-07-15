import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch import Tensor
from torch.nn import Sequential, Linear, Sigmoid, Parameter
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.typing import OptTensor

from models.gcn_drop import DropGCN, GNNLayer, BbGCN


class MlpDropGCN(DropGCN):
    """Perform *DropMessage* on GCN with **rate** predicted from an MLP model"""

    def __init__(self, feature_num: int, hidden_num: int, output_num: int):
        super().__init__(feature_num=feature_num,
                         output_num=output_num)
        self.gnn1 = AdaptiveGNNLayer(feature_num, hidden_num)
        self.gnn2 = GNNLayer(hidden_num, output_num)


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
        self.mlp = Linear(3 * hidden_channels, 1)
        self.multi_dropout = MultiDropout()

    def message(self,
                x_i: Tensor,
                x_j: Tensor,
                edge_index_target: Tensor,
                dim_size: int):
        # if not self.training:
        #     return x_j

        x_i_transformed = self.pre_transform_linear(x_i)
        x_j_transformed = self.pre_transform_linear(x_j)
        diff = x_i_transformed - x_j_transformed
        x_cat = torch.cat([x_i_transformed, x_j_transformed, diff], dim=1)
        similarity_exp = torch.exp(self.mlp(x_cat))
        neighbor_sum = (similarity_exp.new_zeros(dim_size, 1).
                        scatter_add(dim=0,
                                    index=edge_index_target.view(-1, similarity_exp.shape[1]),
                                    src=similarity_exp))

        drop_rate_mlp = similarity_exp / neighbor_sum[edge_index_target]
        print(f"MLP output: {drop_rate_mlp[:3]}")
        # print("Dropping...")

        print(f"Before drop: {x_j[:3]}")

        # drop messages
        # x_j = _multi_dropout(x_j, probability=drop_rate_mlp)
        x_j = self.multi_dropout(x_j, p=drop_rate_mlp)

        print(f"After drop: {x_j[:3]}")
        # print("Dropped.")

        return x_j


def _multi_dropout(x: Tensor, probability: Tensor) -> Tensor:
    assert x.shape[0] == probability.shape[0]
    mask: Tensor = torch.rand_like(x) > probability
    return mask * x  # / (1.0 - probability)


class MultiDropout(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask: Tensor | None = None

    def forward(self, x: Tensor, p: Tensor) -> Tensor:
        if self.training:
            assert x.shape[0] == p.shape[0]
            self.mask = torch.rand_like(x) > p
            return x * self.mask
        else:
            return x * (1.0 - p)

    def backward(self, y: Tensor):
        print("Backward")
        return y * self.mask
