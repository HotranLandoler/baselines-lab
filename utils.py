import random
from argparse import Namespace

import numpy as np
import torch
import torch_geometric.transforms as transforms
from torch.nn import Module
from torch_geometric.data import Data

import data_processing
from models import GCN, DropGCN, MlpDropGCN


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

    data = _prepare_data()
    model = _prepare_model(args, data)
    _process_data(args, data)
    return data.to(args.device), model.to(args.device)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_data() -> Data:
    dataset_transform = transforms.Compose([transforms.ToSparseTensor()])
    dataset = data_processing.DGraphDataset(transform=dataset_transform)
    return dataset[0]


def _prepare_model(args: Namespace, data: Data) -> Module:
    match args.model:
        case "gcn":
            model = GCN(in_channels=data.num_features, hidden_channels=args.hidden_size,
                        out_channels=args.num_classes, dropout=args.dropout,
                        num_layers=args.num_layers)
        case "dropgcn":
            model = DropGCN(feature_num=data.num_features,
                            output_num=args.num_classes)
        case "mlpdropgcn":
            model = MlpDropGCN(feature_num=data.num_features,
                               hidden_num=args.hidden_size,
                               output_num=args.num_classes)
        case _:
            raise NotImplementedError(f"Model {args.model} not implemented")

    assert model is not None
    return model


def _process_data(args: Namespace, data: Data):
    match args.model:
        case "tgat":
            data_processing.process_tgat_data(data)
            raise NotImplementedError(f"Model {args.model} not implemented")
        case _:
            data_processing.data_preprocess(data)
