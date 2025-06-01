import os
import pickle
import random
from typing import Optional, Callable, Literal

import numpy as np
import torch
import pandas as pd
import torch_geometric.transforms
import torch_geometric.utils as pyg_utils
import sklearn.model_selection
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import InMemoryDataset, TemporalData
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, JODIEDataset
from torch_geometric.transforms import BaseTransform

import utils
from models.graph_smote import GraphSmote

DATA_ROOT = "./data/"
RAW_DIR = "raw"
PROCESSED_DIR = "processed"


def get_cora(transform: BaseTransform | None) -> Data:
    """Get the Cora dataset."""
    dataset = Planetoid(root=DATA_ROOT,
                        name="Cora",
                        transform=transform)
    return dataset[0]


def get_jodie(name: Literal["Reddit", "Wikipedia"],
              transform: BaseTransform | None) -> TemporalData:
    """Get the JODIE(Reddit or Wiki) dataset."""
    dataset = JODIEDataset(root=DATA_ROOT,
                           name=name,
                           transform=transform)
    return dataset[0]


def get_yelp() -> Data:
    print("Loading yelp from yelp.dat...")
    return pickle.load(open(f'{DATA_ROOT}yelp.dat', 'rb'))


def get_elliptic() -> Data:
    print("Loading Elliptic dataset...")
    return pickle.load(open(f'{DATA_ROOT}elliptic.dat', 'rb'))


def process_dgraph(data: Data, max_time_steps=32, is_model_dagad=False) -> Data:
    """Perform pre-processing on DGraph before training.

    - Normalize features(x)
    - Set edge-time, node-time, node-out-degree, mean-node-out-time-interval
    - To Undirected Graph
    """
    # Get Edge-time
    data.edge_time = data.edge_time - data.edge_time.min()  # process edge time
    data.edge_time = data.edge_time / data.edge_time.max()
    data.edge_time = (data.edge_time * max_time_steps).long()
    data.edge_time = data.edge_time.view(-1, 1).float()

    # Get Node-time
    edge_index_reshaped = torch.stack((data.edge_index[0],
                                       data.edge_index[1]))
    edge = torch.cat([edge_index_reshaped, data.edge_time.view(1, -1)], dim=0)  # process node time
    degree = pd.DataFrame(edge.T.numpy()).groupby(0).min().values
    ids = pd.DataFrame(edge_index_reshaped.T.numpy()).groupby(0).count().index.values
    key = {i: 0 for i in range(data.x.shape[0])}
    for i in range(len(ids)):
        key[ids[i]] = degree[i][1]
    node_time = np.array(list(key.values()))
    data.node_time = torch.tensor(node_time, dtype=torch.float)

    # Normalization
    x: Tensor = data.x
    x[x == -1.] = 0.
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

    # Get Node-out-degree
    data.node_out_degree = torch_geometric.utils.degree(
        data.edge_index[0], num_nodes=data.num_nodes).reshape(-1, 1)

    data.node_in_degree = torch_geometric.utils.degree(
        data.edge_index[1], num_nodes=data.num_nodes).reshape(-1, 1)

    # Get average node-out-time-interval
    node_out_times = pd.DataFrame(
        np.concatenate(
            (data.edge_index[0].reshape(-1, 1), data.edge_time.int().reshape(-1, 1)), axis=-1),
        columns=["node_out", "time"])
    edge_mean_out_time_interval = node_out_times.groupby("node_out").agg(_get_mean_out_time_interval)
    node_mean_out_time_interval = np.zeros(data.num_nodes)
    node_mean_out_time_interval[edge_mean_out_time_interval.index] = edge_mean_out_time_interval.values.flatten()
    data.node_mean_out_time_interval = torch.tensor(node_mean_out_time_interval.reshape(-1, 1),
                                                    dtype=data.edge_time.dtype)

    # trans to undirected graph
    data.edge_index = torch.cat((data.edge_index, data.edge_index[[1, 0], :]), dim=1)
    data.edge_time = torch.cat((data.edge_time, data.edge_time), dim=0)

    train_subset = False
    if train_subset:
        total_size = data.train_mask.shape[0] + data.val_mask.shape[0] + data.test_mask.shape[0]
        print(f"Original Ratio: {data.train_mask.shape[0] / total_size}% Train mask size: {data.train_mask.shape[0]}")
        train_ratio = 0.1
        subset_size = int(total_size * train_ratio)
        subset_index = torch.randperm(data.train_mask.shape[0])[:subset_size]
        data.train_mask = data.train_mask[subset_index]
        print(f"Ratio: {train_ratio}, Subset size: {subset_size}, Train mask size: {data.train_mask.shape}")

    # Process for DAGAD model
    if is_model_dagad:
        anomaly_mask: Tensor = data.y == 1
        benign_mask: Tensor = data.y == 0

        # Convert index masks to bool masks
        data.train_mask = pyg_utils.index_to_mask(data.train_mask, data.num_nodes)
        data.val_mask = pyg_utils.index_to_mask(data.val_mask, data.num_nodes)
        data.test_mask = pyg_utils.index_to_mask(data.test_mask, data.num_nodes)

        data.train_anm = data.train_mask * anomaly_mask
        data.train_norm = data.train_mask * benign_mask

    data.temporal_embedding = utils.load_tpa_embedding("DGraph")

    return data


def process_ethereum(data: Data, max_time_steps=32, train_ratio=0.4, test_ratio=0.67) -> Data:
    """Perform pre-processing on Ethereum before training.
    """
    # Get Edge-time
    data.edge_time = data.edge_time - data.edge_time.min()  # process edge time
    data.edge_time = data.edge_time / data.edge_time.max()
    data.edge_time = (data.edge_time * max_time_steps).long()
    data.edge_time = data.edge_time.view(-1, 1).float()

    # Get Node-time
    edge_index_reshaped = torch.stack((data.edge_index[0],
                                       data.edge_index[1]))
    edge = torch.cat([edge_index_reshaped, data.edge_time.view(1, -1)], dim=0)  # process node time
    degree = pd.DataFrame(edge.T.numpy()).groupby(0).min().values
    ids = pd.DataFrame(edge_index_reshaped.T.numpy()).groupby(0).count().index.values
    key = {i: 0 for i in range(data.num_nodes)}
    for i in range(len(ids)):
        key[ids[i]] = degree[i][1]
    node_time = np.array(list(key.values()))
    data.node_time = torch.tensor(node_time, dtype=torch.float)

    # Get Node-out-degree
    data.node_out_degree = torch_geometric.utils.degree(
        data.edge_index[0], num_nodes=data.num_nodes).reshape(-1, 1)

    # Normalization
    deg = torch_geometric.utils.degree(
        data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
    deg[deg > 16] = 16
    x = torch_geometric.utils.one_hot(deg, num_classes=16 + 1, dtype=torch.float)
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

    # Get average node-out-time-interval
    node_out_times = pd.DataFrame(
        np.concatenate(
            (data.edge_index[0].reshape(-1, 1), data.edge_time.int().reshape(-1, 1)), axis=-1),
        columns=["node_out", "time"])
    edge_mean_out_time_interval = node_out_times.groupby("node_out").agg(_get_mean_out_time_interval)
    node_mean_out_time_interval = np.zeros(data.num_nodes)
    node_mean_out_time_interval[edge_mean_out_time_interval.index] = edge_mean_out_time_interval.values.flatten()
    data.node_mean_out_time_interval = torch.tensor(node_mean_out_time_interval.reshape(-1, 1),
                                                    dtype=data.edge_time.dtype)

    # trans to undirected graph
    data.edge_index = torch.cat((data.edge_index, data.edge_index[[1, 0], :]), dim=1)
    data.edge_time = torch.cat((data.edge_time, data.edge_time), dim=0)

    # Train-test split
    indexes = list(range(data.num_nodes))

    train_mask, rest_indexes, _, rest_y = sklearn.model_selection.train_test_split(
        indexes, data.y, stratify=data.y, train_size=train_ratio,
        random_state=2, shuffle=True
    )
    val_mask, test_mask, _, _ = sklearn.model_selection.train_test_split(
        rest_indexes, rest_y, stratify=rest_y, test_size=test_ratio,
        random_state=2, shuffle=True
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def process_jodie(data: TemporalData) -> Data:
    x = torch.zeros((data.num_nodes, data.msg.shape[1]))
    # x = torch.arange(0, data.num_nodes, dtype=torch.float).reshape(-1, 1)
    edge_index = torch.stack((data.src, data.dst))
    edge_time = data.t.view(-1, 1)

    y = torch.zeros(data.num_nodes, dtype=data.y.dtype)
    y[data.src[data.y == 1]] = 1

    edge_msg = torch.cat([edge_index.T, data.msg], dim=-1)
    msg_mean = torch.tensor(pd.DataFrame(edge_msg.numpy()).groupby(0).mean().values[:, 1:])
    # x = torch.zeros((data.num_nodes - msg_mean.shape[0], data.msg.shape[1]))
    # x = torch.cat([msg_mean, x])

    edge = torch.cat([edge_index, edge_time.view(1, -1)])
    degree = pd.DataFrame(edge.T.numpy()).groupby(0).min().values
    ids = pd.DataFrame(edge_index.T.numpy()).groupby(0).count().index.values
    key = {i: 0 for i in range(data.num_nodes)}
    for i in range(len(ids)):
        key[ids[i]] = degree[i][1]
    node_time = torch.tensor(list(key.values()))

    node_out_degree = torch_geometric.utils.degree(
        edge_index[0], num_nodes=data.num_nodes).reshape(-1, 1)

    indexes = [i for i in range(8227)]
    random.shuffle(indexes)
    train_mask = torch.tensor(indexes[:5759])
    val_mask = torch.tensor(indexes[5759:6993])
    test_mask = torch.tensor(indexes[6993:])

    return Data(x, edge_index, data.msg, y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                edge_time=edge_time, node_time=node_time,
                node_out_degree=node_out_degree)


def process_yelpchi(data: Data, train_ratio=0.4, test_ratio=0.67) -> Data:
    edge_time = torch.zeros((data.num_edges, 1), dtype=torch.float32)
    node_time = torch.zeros(data.num_nodes, dtype=torch.float32)
    node_mean_out_time_interval = torch.zeros((data.num_nodes, 1), dtype=torch.float32)

    if not hasattr(data, 'adj_t'):
        node_out_degree = torch_geometric.utils.degree(
            data.edge_index[0], num_nodes=data.num_nodes).reshape(-1, 1)
        data.node_out_degree = (node_out_degree - node_out_degree.mean(0)) / node_out_degree.std(0)

    data.edge_time = edge_time
    data.node_time = node_time
    data.node_mean_out_time_interval = node_mean_out_time_interval

    data.train_mask = pyg_utils.mask_to_index(data.train_mask)
    data.val_mask = pyg_utils.mask_to_index(data.val_mask)
    data.test_mask = pyg_utils.mask_to_index(data.test_mask)

    return data


def process_elliptic(data: Data) -> Data:
    x: Tensor = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
    data = process_yelpchi(data)
    return data


def _get_mean_out_time_interval(series: pd.core.series.Series) -> pd.core.series.Series:
    if series.shape[0] <= 1:
        return 0
    return np.diff(np.sort(series)).mean()
