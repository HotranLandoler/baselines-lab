import os
from typing import Optional, Callable

import numpy as np
import scipy
import torch
from torch_geometric.data import InMemoryDataset, Data

import data_processing


class YelpChiDataset(InMemoryDataset):
    def __init__(self, root=data_processing.DATA_ROOT,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = "YelpChi"
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
        return ['YelpChi.mat']

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = [self._read_yelpchi()]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _read_yelpchi(self) -> Data:
        print('Reading YelpChi...')
        raw_name = self.raw_file_names[0]
        yelp: dict = scipy.io.loadmat(os.path.join(self.raw_dir, raw_name))

        x = yelp['features'].todense().A
        y = yelp['label'].flatten()
        adj = yelp['homo'].tocoo()

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.int64)
        edge_index = torch.tensor(np.stack([adj.row, adj.col]), dtype=torch.int64)

        data = Data(x=x, edge_index=edge_index, y=y)

        return data
