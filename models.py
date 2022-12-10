import torch.nn
from torch import Tensor
from cogdl.data import Graph
from cogdl.layers import GCNLayer
from cogdl.models import BaseModel


class GCN(BaseModel):
    """GCN Model implemented based on Cogdl library"""
    def __init__(self, in_feats: int, out_feats: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [
                GCNLayer(
                    shapes[i],
                    shapes[i + 1],
                    dropout=dropout if i != num_layers - 1 else 0,
                    activation="relu" if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            in_feats=args.num_features,
            hidden_size=args.hidden_size,
            out_feats=args.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    def forward(self, graph: Graph) -> Tensor:
        graph.sym_norm()
        h: Tensor = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
        return h.log_softmax(dim=-1)

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()
