import os

import numpy as np
import torch
import cogdl.utils.graph_utils as cog_utils
import pandas as pd
from torch import Tensor
from cogdl.data import Graph
from cogdl.datasets import NodeDataset

RAW_PATH = "data/DGraph/raw/dgraphfin.npz"
PROCESSED_DIR = "data/DGraph/processed/"
PROCESSED_NAME = "dgraph.pt"


class DGraphDataset(NodeDataset):
    # @property
    # def raw_file_names(self):
    #     pass
    #
    # @property
    # def processed_file_names(self):
    #     pass

    def __init__(self, path=PROCESSED_DIR + PROCESSED_NAME, to_undirected=True):
        # Make processed dir
        if not os.path.exists(PROCESSED_DIR):
            os.mkdir(PROCESSED_DIR)
        self.path = path
        self.to_undirected = to_undirected
        super(DGraphDataset, self).__init__(path, scale_feat=False, metric="accuracy")

    def process(self):
        """
        Load DGraph dataset and transform to `Graph`,
        runs only when no processed file is found.
        """
        print("Processing Data...")
        data_file = np.load(RAW_PATH)

        edge_index = torch.from_numpy(data_file['edge_index']).transpose(0, 1)
        if self.to_undirected:
            # To undirected
            edge_index = cog_utils.to_undirected(edge_index)
        x = torch.from_numpy(data_file['x']).float()
        y = torch.from_numpy(data_file['y'])

        # set train/val/test mask in node_classification task
        train_mask = torch.zeros(x.shape[0]).bool()
        train_mask[data_file['train_mask']] = True

        val_mask = torch.zeros(x.shape[0]).bool()
        val_mask[data_file['valid_mask']] = True

        test_mask = torch.zeros(x.shape[0]).bool()
        test_mask[data_file['test_mask']] = True

        edge_time = torch.tensor(data_file['edge_timestamp']).long()

        # Cogdl Graph does not support adding custom attribute starting with 'edge_'
        data = DynamicGraph(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                            val_mask=val_mask, test_mask=test_mask, edge_time=edge_time)
        # data.edge_time = edge_time
        return data


class DynamicGraph(Graph):
    def __init__(self, x: Tensor, y: Tensor, edge_time: Tensor, **kwargs):
        super().__init__(x, y, **kwargs)
        self.edge_time = edge_time

    def to(self, device, *keys):
        self.edge_time = self.edge_time.to(device)
        return super().to(device, *keys)

    def __setattr__(self, key, value):
        if key == 'edge_time':
            self.__dict__[key] = value
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key):
        if key == 'edge_time':
            return getattr(self, key)
        return super().__getitem__(key)


def data_preprocess(dataset: DynamicGraph):
    """Perform pre-processing on dataset before training"""
    # Normalization
    x = dataset.x
    x = (x - x.mean(0)) / x.std(0)
    dataset.x = x


def __build_tg_data():
    """Data for TGAT"""
    origin_data = np.load(RAW_PATH)
    data = Graph()
    data.x = torch.tensor(origin_data['x']).float()
    data.y = torch.tensor(origin_data['y']).long()
    data.edge_index = torch.tensor(origin_data['edge_index']).long().T
    data.train_mask = torch.tensor(origin_data['train_mask']).long()
    data.val_mask = torch.tensor(origin_data['valid_mask']).long()
    data.test_mask = torch.tensor(origin_data['test_mask']).long()
    data.edge_time = torch.tensor(origin_data['edge_timestamp']).long()
    data.edge_index = cog_utils.to_undirected(data.edge_index)
    return data


def process_tgat_data(data: DynamicGraph, max_time_steps=32):
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

    # trans to undirected graph
    undirected_edge = torch.cat((edge_index_reshaped, edge_index_reshaped[[1, 0], :]), dim=1)
    data.edge_index = (undirected_edge[0, :], undirected_edge[1, :])
    data.edge_time = torch.cat((data.edge_time, data.edge_time), dim=0)

    return data
