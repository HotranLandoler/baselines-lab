import random
import math

import torch.nn
import torch.nn.functional as F
import numpy as np
import scipy.spatial.distance as sp_distance
import torch_sparse
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import SAGEConv
from torch_sparse import SparseTensor


class GraphSmote(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor, idx_train: Tensor):
        x, y_new, idx_train_new = self.recon_upsample(x, y, idx_train)
        return x, y_new, idx_train_new

    @staticmethod
    def recon_upsample(embed, labels, idx_train: Tensor, edge_index: Tensor, edge_time: Tensor, node_time: Tensor,
                       max_time_steps: int, adj: SparseTensor = None, portion=1.0, im_class_num=1):
        """SMOTE"""
        c_largest = 1
        avg_number = int(idx_train.shape[0] / (c_largest + 1))
        # ipdb.set_trace()
        adj_new = None

        # 训练集中正常类节点index
        idx_benign_train: Tensor = idx_train[(labels == 0)[idx_train]]

        for i in range(im_class_num):
            # 训练集中少数类节点index
            chosen = idx_train[(labels == (c_largest - i))[idx_train]]
            num = int(chosen.shape[0] * portion)
            if portion == 0:
                c_portion = int(avg_number / chosen.shape[0])
                num = chosen.shape[0]
            else:
                c_portion = 1

            for _ in range(c_portion):
                chosen = chosen[:num]

                chosen_embed = embed[chosen, :]

                # 获得两两节点间嵌入距离矩阵
                distance = sp_distance.squareform(sp_distance.pdist(chosen_embed.cpu().detach()))
                np.fill_diagonal(distance, distance.max() + 100)

                # 为每个节点找到嵌入最接近的节点index
                idx_neighbor = distance.argmin(axis=-1)

                # 计算新节点嵌入
                interp_place = random.random()
                new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

                new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
                idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
                idx_train_append = idx_train.new(idx_new)

                embed = torch.cat((embed, new_embed), 0)
                labels = torch.cat((labels, new_labels), 0)
                idx_train = torch.cat((idx_train, idx_train_append), 0)

                if adj is not None:
                    if adj_new is None:
                        # adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                        adj_new = adj[chosen, :] + adj[idx_neighbor, :]
                    else:
                        temp = adj.new(torch.clamp_(adj[chosen, :] + adj[idx_neighbor, :], min=0.0, max=1.0))
                        adj_new = torch.cat((adj_new, temp), 0)

        # Set edge index
        idx_benign_sampled = idx_benign_train[
            torch.randint_like(idx_train_append, low=0, high=idx_benign_train.shape[0])]
        edge_index_append = torch.stack((idx_benign_sampled, idx_train_append), dim=0)
        # new_to_chosen_edge_index = torch.stack((idx_train_append, chosen), dim=0)
        # new_to_neighbor_edge_index = torch.stack((idx_train_append,
        #                                           torch.tensor(idx_neighbor, dtype=torch.int64)), dim=0)
        # edge_index_append = torch.cat((new_to_chosen_edge_index, new_to_neighbor_edge_index), dim=-1)
        edge_index_new = torch.cat((edge_index, edge_index_append), dim=-1)

        # Set Edge Time
        edge_time_append = node_time[idx_benign_sampled].reshape(-1, 1)
        # chosen_node_time = node_time[chosen]
        # neighbor_node_time = node_time[idx_neighbor]
        # edge_time_append = torch.cat((chosen_node_time, neighbor_node_time), dim=0).reshape(-1, 1)

        # edge_time_append = torch.zeros(size=(edge_index_append.shape[1], 1), dtype=edge_time.dtype)
        # edge_time_append: Tensor = torch.rand(size=(edge_index_append.shape[1], 1), dtype=edge_time.dtype) * max_time_steps

        edge_time_new = torch.cat((edge_time, edge_time_append), dim=0)

        # node_time_stack = torch.stack((chosen_node_time, neighbor_node_time), dim=0)
        # node_time_append, _ = torch.min(node_time_stack, dim=0)
        node_time_append = node_time[idx_benign_sampled]
        node_time_new = torch.cat((node_time, node_time_append), dim=0)

        if adj is not None:
            # add_num = adj_new.shape[0]
            add_num = chosen.shape[0]
            # new_adj = adj.new(torch.Size((adj.shape[0] + add_num, adj.shape[0] + add_num))).fill_(0.0)
            adj_shape_0 = adj.storage.sparse_sizes()[0]
            new_adj = SparseTensor(row=adj.storage.row(),
                                   col=adj.storage.col(),
                                   sparse_sizes=(adj.storage.sparse_sizes()[0] + add_num,
                                                 adj.storage.sparse_sizes()[0] + add_num))
            new_adj[:adj_shape_0, :adj_shape_0] = adj[:, :]
            new_adj[adj_shape_0:, :adj_shape_0] = adj_new[:, :]
            new_adj[:adj_shape_0, adj_shape_0:] = torch.transpose(adj_new, 0, 1)[:, :]

            return embed, labels, idx_train, new_adj.detach()

        else:
            # return embed, labels, idx_train, edge_index, edge_time, node_time
            return embed, labels, idx_train, edge_index_new, edge_time_new, node_time_new


class SageEncoder(torch.nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super().__init__()

        self.sage1 = SAGEConv(nfeat, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Decoder(torch.nn.Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))

        return adj_out


class Classifier(torch.nn.Module):
    def __init__(self, nhid: int, nclass: int):
        super(Classifier, self).__init__()
        self.mlp = torch.nn.Linear(nhid, nclass)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        out = self.mlp(x)
        out = F.log_softmax(out, dim=-1)
        return out
