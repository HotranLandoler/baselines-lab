import numpy as np
import torch
import cogdl.utils.graph_utils as cog_utils
from cogdl.data import Graph
from cogdl.datasets import NodeDataset


class DGraphDataset(NodeDataset):
    # @property
    # def raw_file_names(self):
    #     pass
    #
    # @property
    # def processed_file_names(self):
    #     pass

    def __init__(self, path="data/DGraph/processed/dgraph.pt"):
        self.path = path
        super(DGraphDataset, self).__init__(path, scale_feat=False, metric="accuracy")

    def process(self):
        """Load DGraph dataset and transform to `Graph`"""
        print("Processing Data...")
        data_file = np.load("data/DGraph/raw/dgraphfin.npz")

        edge_index = torch.from_numpy(data_file['edge_index']).transpose(0, 1)
        x = torch.from_numpy(data_file['x']).float()
        y = torch.from_numpy(data_file['y'])

        # set train/val/test mask in node_classification task
        train_mask = torch.zeros(x.shape[0]).bool()
        train_mask[data_file['train_mask']] = True

        val_mask = torch.zeros(x.shape[0]).bool()
        val_mask[data_file['valid_mask']] = True

        test_mask = torch.zeros(x.shape[0]).bool()
        test_mask[data_file['test_mask']] = True

        data = Graph(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                     val_mask=val_mask, test_mask=test_mask)
        return data


def data_preprocess(dataset: Graph):
    """Perform pre-processing on dataset before training"""
    # Normalization
    x = dataset.x
    x = (x - x.mean(0)) / x.std(0)
    dataset.x = x

    # To undirected
    dataset.edge_index = cog_utils.to_undirected(dataset.edge_index)
