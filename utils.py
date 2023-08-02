import random
from argparse import Namespace

import numpy as np
import torch
import torch_geometric.transforms as transforms
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data

import data_processing
import datasets
from models import GCN, DropGCN, MlpDropGCN, H2GCN_EGO, MLP, TGAT, GraphSAGE, AMNet


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
    if args.model in ["tgat", "amnet"]:
        # dataset_transform = transforms.OneHotDegree(max_degree=10)
        dataset_transform = None
    else:
        dataset_transform = transforms.Compose([transforms.ToSparseTensor()])

    match args.dataset:
        case "DGraph":
            data = datasets.DGraphDataset(transform=dataset_transform)[0]
        case "Cora":
            data = data_processing.get_cora(transform=dataset_transform)
        case "Reddit" | "Wikipedia":
            data = data_processing.get_jodie(args.dataset, transform=dataset_transform)
        case "Yelp":
            data = datasets.YelpChiDataset(transform=dataset_transform)[0]
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
                            output_num=args.num_classes)
        case "mlpdropgcn":
            model = MlpDropGCN(feature_num=data.num_features,
                               hidden_num=args.hidden_size,
                               output_num=args.num_classes)
        case "h2gcn":
            # model = H2GCN(feat_dim=data.num_features,
            #               hidden_dim=args.hidden_size,
            #               class_dim=args.num_classes,
            #               dropout=args.dropout)
            model = H2GCN_EGO(in_channels=data.num_features, hidden_channels=args.hidden_size,
                              out_channels=args.num_classes, dropout=args.dropout,
                              num_layers=args.num_layers)
        case "mlp":
            model = MLP(in_channels=data.num_features,
                        hidden_channels=args.hidden_size,
                        out_channels=args.num_classes)
        case "tgat":
            model = TGAT(in_channels=data.num_features, out_channels=args.num_classes)
        case "amnet":
            model = AMNet(in_channels=data.num_features,
                          hid_channels=args.hidden_size,
                          num_class=args.num_classes,
                          K=7,
                          filter_num=2)
        case _:
            raise NotImplementedError(f"Model {args.model} not implemented")

    assert model is not None
    return model


def _process_data(args: Namespace, data: Data) -> Data:
    if args.dataset == "DGraph" and args.model != "tgat":
        return data_processing.data_preprocess(data)

    if args.dataset == "Yelp":
        return data_processing.process_yelpchi(data)

    # TODO
    if args.dataset == "Wikipedia":
        return data_processing.process_tgat_data(args.dataset, data)

    match args.model:
        case "tgat":
            return data_processing.process_tgat_data(args.dataset, data)
        case _:
            return data
