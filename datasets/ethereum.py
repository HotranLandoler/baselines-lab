import os
import pickle
from typing import Optional, Callable

import networkx as nx
import numpy as np
import torch
from networkx.classes.multidigraph import MultiDiGraph
from torch_geometric.data import InMemoryDataset, Data

import data_processing


class EthereumDataset(InMemoryDataset):
    """https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network"""
    def __init__(self, root=data_processing.DATA_ROOT,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = "Ethereum"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            map_location=torch.device('cpu'))

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, data_processing.RAW_DIR)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, data_processing.PROCESSED_DIR)

    @property
    def raw_file_names(self) -> list[str]:
        return ['MulDiGraph.pkl']

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = [self._read_ethereum()]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _read_ethereum(self) -> Data:
        print('Reading Ethereum...')
        raw_name = self.raw_file_names[0]
        graph = _load_pickle(os.path.join(self.raw_dir, raw_name))

        adj = nx.to_scipy_sparse_array(graph, format="coo")
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        y = [graph.nodes[node]['isp'] for _, node in enumerate(nx.nodes(graph))]
        edge_time = [graph[u][v][0]['timestamp'] for _, (u, v) in enumerate(nx.edges(graph))]

        y = torch.tensor(y, dtype=torch.int64)
        edge_time = torch.tensor(edge_time, dtype=torch.int64)

        data = Data(y=y, edge_index=edge_index, edge_time=edge_time, num_nodes=y.shape[0])

        return data


def _load_pickle(file_name: str) -> MultiDiGraph:
    with open(file_name, 'rb') as f:
        return pickle.load(f)
