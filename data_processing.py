import os
from typing import Optional, Callable

import numpy as np
import torch
import pandas as pd
import torch_geometric.transforms
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import BaseTransform

DATA_ROOT = "./data/"
RAW_DIR = "raw"
PROCESSED_DIR = "processed"


class DGraphDataset(InMemoryDataset):
    """DGraphFin Dataset"""
    def __init__(self, root=DATA_ROOT, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = "DGraph"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            map_location=torch.device('cpu'))

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, RAW_DIR)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, PROCESSED_DIR)

    @property
    def raw_file_names(self) -> list[str]:
        return ['dgraphfin.npz']

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = [self._read_dgraphfin()]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _read_dgraphfin(self) -> Data:
        print('Reading DGraph...')
        raw_name = self.raw_file_names[0]
        item: dict = np.load(os.path.join(self.raw_dir, raw_name))

        x = item['x']
        # y = item['y'].reshape(-1, 1)
        y = item['y']
        edge_index = item['edge_index']
        edge_type = item['edge_type']
        train_mask = item['train_mask']
        val_mask = item['valid_mask']
        test_mask = item['test_mask']
        edge_time = item['edge_timestamp']

        x = torch.tensor(x, dtype=torch.float).contiguous()
        y = torch.tensor(y, dtype=torch.int64)
        edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.float)
        train_mask = torch.tensor(train_mask, dtype=torch.int64)
        val_mask = torch.tensor(val_mask, dtype=torch.int64)
        test_mask = torch.tensor(test_mask, dtype=torch.int64)
        edge_time = torch.tensor(edge_time, dtype=torch.int64)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.edge_time = edge_time

        return data


def get_cora(transform: BaseTransform) -> Data:
    """Get the Cora dataset."""
    dataset = Planetoid(root=DATA_ROOT,
                        name="Cora",
                        transform=transform)
    return dataset[0]


def data_preprocess(data: Data) -> Data:
    """Perform pre-processing on dataset before training.

    - To Undirected Graph
    - Normalize features(x)
    - Reshape y
    """
    # To undirected
    data.adj_t = data.adj_t.to_symmetric()
    # Normalization
    x: Tensor = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
    # Reshape y
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    return data


# def __build_tg_data():
#     """Data for TGAT"""
#     origin_data = np.load(RAW_PATH)
#     data = Graph()
#     data.x = torch.tensor(origin_data['x']).float()
#     data.y = torch.tensor(origin_data['y']).long()
#     data.edge_index = torch.tensor(origin_data['edge_index']).long().T
#     data.train_mask = torch.tensor(origin_data['train_mask']).long()
#     data.val_mask = torch.tensor(origin_data['valid_mask']).long()
#     data.test_mask = torch.tensor(origin_data['test_mask']).long()
#     data.edge_time = torch.tensor(origin_data['edge_timestamp']).long()
#     data.edge_index = cog_utils.to_undirected(data.edge_index)
#     return data


def process_tgat_data(data: Data, max_time_steps=32):
    """https://github.com/hxttkl/DGraph_Experiments"""
    data.edge_time = data.edge_time - data.edge_time.min()  # process edge time
    data.edge_time = data.edge_time / data.edge_time.max()
    data.edge_time = (data.edge_time * max_time_steps).long()
    data.edge_time = data.edge_time.view(-1, 1).float()

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

    data.node_out_degree = torch_geometric.utils.degree(
        data.edge_index[0], num_nodes=data.num_nodes).reshape(-1, 1)

    # trans to undirected graph
    data.edge_index = torch.cat((data.edge_index, data.edge_index[[1, 0], :]), dim=1)
    data.edge_time = torch.cat((data.edge_time, data.edge_time), dim=0)

    return data
