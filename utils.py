import pickle
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms
import torch_geometric.transforms as transforms
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data

import data_processing
import datasets
import models.h2gcn.utils
from models import GCN, DropGCN, H2GCN, H2GCN_EGO, MLP, TGAT, GraphSAGE, DropSAGE, AMNet, GFCA
from models.dagad import DAGAD


def load_tpa_embedding(dataset_name: str) -> Tensor:
    path = Path("data") / f"TPA-{dataset_name}.embedding"
    return torch.load(path, pickle_module=pickle)

def prepare_data_and_model(args: Namespace) -> tuple[Data, Module]:
    """Get processed data and model based on training setting.

    - Get data
    - Get Model
    - Process data
    - Move data and model to device

    Args:
        args: Training arguments.

    Returns:
        Tuple(Processed data, model).

    Raises:
        ValueError: `args.device` not available.
        NotImplementedError: The model in args is not implemented.
    """
    if "cuda" in args.device and not torch.cuda.is_available():
        raise ValueError("Device CUDA not available")

    data = _prepare_data(args)
    data = _process_data(args, data)
    model = _prepare_model(args, data)
    return data.to(args.device), model.to(args.device)


def get_loss_weight(args: Namespace) -> Tensor | None:
    match args.dataset:
        case "DGraph" | "Wikipedia" | "Yelp":
            return torch.tensor([1, args.loss_weight],
                                dtype=torch.float32,
                                device=args.device)
        case _:
            return None


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_data(args: Namespace) -> Data:
    dataset_transform = None

    match args.dataset:
        case "DGraph":
            data = datasets.DGraphDataset(transform=dataset_transform)[0]
        case "Cora":
            data = data_processing.get_cora(transform=dataset_transform)
        case "Reddit" | "Wikipedia":
            data = data_processing.get_jodie(args.dataset, transform=dataset_transform)
        case "Yelp":
            # data = datasets.YelpChiDataset(transform=dataset_transform)[0]
            data = data_processing.get_yelp()
        case "Elliptic":
            data = data_processing.get_elliptic()
        case "Ethereum":
            # onehot_degree = torch_geometric.transforms.OneHotDegree(max_degree=16, cat=False)
            data = datasets.EthereumDataset(transform=dataset_transform)[0]
        case _:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    return data


def _prepare_model(args: Namespace, data: Data) -> Module:
    match args.model:
        case "gcn":
            model = GCN(in_channels=data.num_features, hidden_channels=args.hidden_size,
                        out_channels=args.num_classes, dropout=args.dropout,
                        num_layers=args.num_layers)
        case "sage":
            model = GraphSAGE(in_channels=data.num_features,
                              hidden_channels=args.hidden_size,
                              num_layers=args.num_layers,
                              out_channels=args.num_classes,
                              dropout=args.dropout)
        case "dropgcn":
            model = DropGCN(feature_num=data.num_features,
                            hidden_num=args.hidden_size,
                            output_num=args.num_classes)
        case "dropsage":
            model = DropSAGE(feature_num=data.num_features,
                             hidden_num=args.hidden_size,
                             output_num=args.num_classes)
        case "h2gcn":
            model = H2GCN(data=data,
                          num_features=data.num_features,
                          num_hidden=args.hidden_size,
                          num_classes=args.num_classes,
                          dropout=args.dropout,
                          device=args.device)
            # model = H2GCN_EGO(in_channels=data.num_features, hidden_channels=args.hidden_size,
            #                   out_channels=args.num_classes, dropout=args.dropout,
            #                   num_layers=args.num_layers)
        case "mlp":
            model = MLP(in_channels=data.num_features,
                        hidden_channels=args.hidden_size,
                        out_channels=args.num_classes)
        case "tgat":
            model = TGAT(in_channels=data.num_features,
                         hid_channels=args.hidden_size,
                         out_channels=args.num_classes)
        case "dagad":
            model = DAGAD(data.num_features, 16, 8, args.num_classes, args.device)
        case "amnet":
            model = AMNet(in_channels=data.num_features,
                          hid_channels=args.hidden_size,
                          num_class=args.num_classes,
                          K=5,
                          filter_num=2)
        case "gfca":
            model = GFCA(in_channels=data.num_features,
                         hid_channels=args.hidden_size,
                         out_channels=args.num_classes,
                         encoding_dim=args.encoding_size,
                         dropout=args.dropout)
        case _:
            raise NotImplementedError(f"Model {args.model} not implemented")

    assert model is not None
    return model


def _process_data(args: Namespace, data: Data) -> Data:
    match args.dataset:
        case "DGraph":
            return data_processing.process_dgraph(data)
        case "Yelp":
            return data_processing.process_yelpchi(data)
        case "Wikipedia":
            return data_processing.process_jodie(data)
        case "Elliptic":
            return data_processing.process_elliptic(data)
        case "Ethereum":
            return data_processing.process_ethereum(data)
        case _:
            return data
