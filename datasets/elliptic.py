from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch_geometric
import torch_geometric.utils as pyg_utils
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset

import data_processing
import options


class Elliptic:
    def __init__(self):
        self.name = Elliptic.__name__
        self._dataset = EllipticBitcoinDatasetWithTime(f"data/{self.name}")

    def process(self) -> Data:
        data = self._dataset[0]

        data.edge_time = data.node_time[data.edge_index[0]].reshape(data.edge_index[0].shape[0], 1)

        data.edge_time = data.edge_time - data.edge_time.min()  # process edge time
        data.edge_time = data.edge_time / data.edge_time.max()
        data.edge_time = (data.edge_time * 49).long()
        data.edge_time = data.edge_time.view(-1, 1).float()

        # Normalization
        x: Tensor = data.x
        x[x == -1.] = 0.
        x = (x - x.mean(0)) / x.std(0)
        data.x = x

        # Get Node-out-degree
        data.node_out_degree = torch_geometric.utils.degree(
            data.edge_index[0], num_nodes=data.num_nodes).reshape(-1, 1)

        node_out_times = pd.DataFrame(
            np.concatenate(
                (data.edge_index[0].reshape(-1, 1), data.edge_time.int().reshape(-1, 1)), axis=-1),
            columns=["node_out", "time"])
        edge_mean_out_time_interval = node_out_times.groupby("node_out").agg(data_processing._get_mean_out_time_interval)
        node_mean_out_time_interval = np.zeros(data.num_nodes)
        node_mean_out_time_interval[edge_mean_out_time_interval.index] = edge_mean_out_time_interval.values.flatten()
        data.node_mean_out_time_interval = torch.tensor(node_mean_out_time_interval.reshape(-1, 1),
                                                        dtype=data.edge_time.dtype)

        data.train_index = pyg_utils.mask_to_index(data.train_mask)
        data.val_index = pyg_utils.mask_to_index(data.val_mask)
        data.test_index = pyg_utils.mask_to_index(data.test_mask)

        return data


class EllipticBitcoinDatasetWithTime(EllipticBitcoinDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)

    def process(self) -> None:
        """Mostly the same with the EllipticBitcoinDataset class"""
        import pandas as pd

        feat_df = pd.read_csv(self.raw_paths[0], header=None)
        edge_df = pd.read_csv(self.raw_paths[1])
        class_df = pd.read_csv(self.raw_paths[2])

        columns = {0: 'txId', 1: 'time_step'}
        feat_df = feat_df.rename(columns=columns)

        feat_df, edge_df, class_df = self._process_df(
            feat_df,
            edge_df,
            class_df,
        )

        x = torch.from_numpy(feat_df.loc[:, 2:].values).to(torch.float)

        # There exists 3 different classes in the dataset:
        # 0=licit,  1=illicit, 2=unknown
        # CHANGE: unknown -> 0
        mapping = {'unknown': 0, '1': 1, '2': 0}
        class_df['class'] = class_df['class'].map(mapping)
        y = torch.from_numpy(class_df['class'].values)

        mapping = {idx: i for i, idx in enumerate(feat_df['txId'].values)}
        edge_df['txId1'] = edge_df['txId1'].map(mapping)
        edge_df['txId2'] = edge_df['txId2'].map(mapping)
        edge_index = torch.from_numpy(edge_df.values).t().contiguous()

        # Timestamp based split:
        # CHANGE train_mask: 1 - 30 time_step, val_mask: 31-35, test_mask: 35-49 time_step
        time_step = torch.from_numpy(feat_df['time_step'].values)
        train_mask = (time_step < 31) & (y != 2)
        val_mask = (time_step < 35) & (y != 2)
        test_mask = (time_step >= 35) & (y != 2)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask, node_time=time_step.to(torch.float))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

# class Elliptic(Dataset):
#     def __init__(self):
#         self.name = Elliptic.__name__
#         self._dataset = pickle.load(open(Path(options.DATA_ROOT) / self.name / "elliptic.dat", 'rb'))
#
#     def process(self) -> Data:
#         data = self._dataset
#         edge_time = torch.zeros((data.num_edges, 1), dtype=torch.float32)
#
#         data.edge_time = edge_time
#
#         data.train_index = pyg_utils.mask_to_index(data.train_mask)
#         data.val_index = pyg_utils.mask_to_index(data.val_mask)
#         data.test_index = pyg_utils.mask_to_index(data.test_mask)
#         return data
