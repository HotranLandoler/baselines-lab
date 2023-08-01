import os
from typing import Optional, Callable

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

import data_processing


class DGraphDataset(InMemoryDataset):
    """DGraphFin Dataset"""
    def __init__(self, root=data_processing.DATA_ROOT,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = "DGraph"
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
