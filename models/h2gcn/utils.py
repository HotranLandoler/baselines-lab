import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.utils as pyg_utils


def edge_index_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
    if device is None:
        device = edge_index.device
    return coo.to(device)


# This function is generated from the following link: https://github.com/EdisonLeeeee/GreatX/blob/master/greatx/utils/modification.py
def remove_edges(edge_index, edges_to_remove):
    edges_to_remove = torch.cat(
            [edges_to_remove, edges_to_remove.flip(0)], dim=1)
    edges_to_remove = edges_to_remove.to(edge_index)

    # it's not intuitive to remove edges from a graph represented as `edge_index`
    edge_weight_remove = torch.zeros(edges_to_remove.size(1)) - 1e5
    edge_weight = torch.cat(
        [torch.ones(edge_index.size(1)), edge_weight_remove], dim=0)
    edge_index = torch.cat([edge_index, edges_to_remove], dim=1).cpu().numpy()
    adj_matrix = sp.csr_matrix(
        (edge_weight.cpu().numpy(), (edge_index[0], edge_index[1])))
    adj_matrix.data[adj_matrix.data < 0] = 0.
    adj_matrix.eliminate_zeros()
    edge_index, _ = pyg_utils.from_scipy_sparse_matrix(adj_matrix)
    return edge_index


def edge_index_to_sparse_tensor_adj(edge_index):
    sparse_adj_adj = pyg_utils.to_scipy_sparse_matrix(edge_index)
    values = sparse_adj_adj.data
    indices = np.vstack((sparse_adj_adj.row, sparse_adj_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_adj_adj.shape
    sparse_adj_adj_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return sparse_adj_adj_tensor


def gcn_norm(edge_index, num_nodes, device):
    a1 = edge_index_to_sparse_tensor_adj(edge_index).to(device)
    d1_adj = torch.diag(pyg_utils.degree(edge_index[0], num_nodes=num_nodes)).to_sparse()
    d1_adj = torch.pow(d1_adj, -0.5).to(device)

    return torch.sparse.mm(torch.sparse.mm(d1_adj, a1), d1_adj)
