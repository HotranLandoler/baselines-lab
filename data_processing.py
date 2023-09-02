import os
import pickle
import random
from typing import Optional, Callable, Literal

import numpy as np
import torch
import pandas as pd
import torch_geometric.transforms
import sklearn.model_selection
from torch import Tensor
from torch_geometric.data import InMemoryDataset, TemporalData
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, JODIEDataset
from torch_geometric.transforms import BaseTransform

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


def process_dgraph(data: Data, max_time_steps=32) -> Data:
    """Perform pre-processing on dataset before training.

    - Normalize features(x)
    - To Undirected Graph
    """
    # Normalization
    x: Tensor = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

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
    data.node_time = torch.tensor(node_time)

    # Get Node-out-degree
    data.node_out_degree = torch_geometric.utils.degree(
        data.edge_index[0], num_nodes=data.num_nodes).reshape(-1, 1)

    # trans to undirected graph
    data.edge_index = torch.cat((data.edge_index, data.edge_index[[1, 0], :]), dim=1)
    data.edge_time = torch.cat((data.edge_time, data.edge_time), dim=0)

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

    # mask = torch.zeros(data.num_nodes, dtype=torch.int)
    # mask[:6459] = 0
    # mask[6459:7844] = 1
    # mask[7844:] = 2
    # # Shuffle
    # indexes = torch.randperm(mask.nelement())
    # train_mask = mask[indexes] == 0
    # val_mask = mask[indexes] == 1
    # test_mask = mask[indexes] == 2

    # train_mask = torch.zeros(data.num_nodes, dtype=torch.int)
    # train_mask[:6459] = True
    # val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # val_mask[6459:7844] = True
    # test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # test_mask[7844:] = True

    # val_time, test_time = list(np.quantile(data.t, [0.70, 0.85]))
    # train_mask = data.t <= val_time
    # val_mask = (data.t > val_time) * (data.t <= test_time)
    # test_mask = data.t > test_time

    return Data(x, edge_index, data.msg, y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                edge_time=edge_time, node_time=node_time,
                node_out_degree=node_out_degree)


def process_yelpchi(data: Data, train_ratio=0.4, test_ratio=0.67) -> Data:
    # indexes = list(range(data.num_nodes))
    #
    # train_mask, rest_indexes, _, rest_y = sklearn.model_selection.train_test_split(
    #     indexes, data.y, stratify=data.y, train_size=train_ratio,
    #     random_state=2, shuffle=True
    # )
    # val_mask, test_mask, _, _ = sklearn.model_selection.train_test_split(
    #     rest_indexes, rest_y, stratify=rest_y, test_size=test_ratio,
    #     random_state=2, shuffle=True
    # )

    edge_time = torch.zeros((data.num_edges, 1), dtype=torch.float32)
    node_time = torch.zeros(data.num_nodes, dtype=torch.float32)

    if not hasattr(data, 'adj_t'):
        node_out_degree = torch_geometric.utils.degree(
            data.edge_index[0], num_nodes=data.num_nodes).reshape(-1, 1)
        data.node_out_degree = (node_out_degree - node_out_degree.mean(0)) / node_out_degree.std(0)

    # data.train_mask = train_mask
    # data.val_mask = val_mask
    # data.test_mask = test_mask

    data.edge_time = edge_time
    data.node_time = node_time

    return data

