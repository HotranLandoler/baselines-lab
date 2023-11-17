import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from torch_geometric.nn.conv import GCNConv

from models import TGAT


class DAGAD(torch.nn.Module):
    """DAGAD_GCN"""
    indices: ndarray

    def __init__(self, input_dim, hidden_dim, fcn_dim, num_classes, device):
        super().__init__()
        self.device = device
        self.fcn_dim = fcn_dim
        self.name = 'DAGAD-GCN'

        # self.GNN_a_conv1 = GCNConv(input_dim, hidden_dim)
        # self.GNN_a_conv2 = GCNConv(hidden_dim, hidden_dim)
        #
        # self.GNN_b_conv1 = GCNConv(input_dim, hidden_dim)
        # self.GNN_b_conv2 = GCNConv(hidden_dim, hidden_dim)

        self.GNN_a_conv1 = TGAT(input_dim, hidden_dim)
        # self.GNN_a_conv2 = TGAT(hidden_dim, hidden_dim)

        self.GNN_b_conv1 = TGAT(input_dim, hidden_dim)
        # self.GNN_b_conv2 = TGAT(hidden_dim, hidden_dim)

        self.fc1_a = nn.Linear(hidden_dim * 2, num_classes)
        # self.fc2_a = nn.Linear(fcn_dim, num_classes)

        self.fc1_b = nn.Linear(hidden_dim * 2, num_classes)
        # self.fc2_b = nn.Linear(fcn_dim, num_classes)

    def forward(self, data, permute=True, **kwargs):
        # h_a = self.GNN_a_conv2(self.GNN_a_conv1(data.x, data.edge_index, data=data).relu(), data.edge_index, data=data).relu()
        # h_b = self.GNN_b_conv2(self.GNN_b_conv1(data.x, data.edge_index, data=data).relu(), data.edge_index, data=data).relu()

        h_a = self.GNN_a_conv1(data.x, data.edge_index, data=data).relu()
        h_b = self.GNN_b_conv1(data.x, data.edge_index, data=data).relu()

        h_back_a = torch.cat((h_a, h_b.detach()), dim=1)
        h_back_b = torch.cat((h_a.detach(), h_b), dim=1)

        h_aug_back_a, h_aug_back_b, data = self._permute_operation(data, h_b, h_a, permute)

        h_back_a = F.relu(h_back_a)
        h_back_b = F.relu(h_back_b)

        h_aug_back_a = F.relu(h_aug_back_a)
        h_aug_back_b = F.relu(h_aug_back_b)

        h_back_a = self.fc1_a(h_back_a)
        # h_back_a = h_back_a.relu()
        h_back_b = self.fc1_b(h_back_b)
        # h_back_b = h_back_b.relu()

        h_aug_back_a = self.fc1_a(h_aug_back_a)
        # h_aug_back_a = h_aug_back_a.relu()
        h_aug_back_b = self.fc1_b(h_aug_back_b)
        # h_aug_back_b = h_aug_back_b.relu()

        # pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)
        # pred_org_back_b = F.log_softmax(self.fc2_b(h_back_b), dim=1)
        #
        # pred_aug_back_a = F.log_softmax(self.fc2_a(h_aug_back_a), dim=1)
        # pred_aug_back_b = F.log_softmax(self.fc2_b(h_aug_back_b), dim=1)

        pred_org_back_a = F.log_softmax(h_back_a, dim=1)
        pred_org_back_b = F.log_softmax(h_back_b, dim=1)

        pred_aug_back_a = F.log_softmax(h_aug_back_a, dim=1)
        pred_aug_back_b = F.log_softmax(h_aug_back_b, dim=1)

        return pred_org_back_a, pred_org_back_b, pred_aug_back_a, pred_aug_back_b, data

    def reset_parameters(self):
        self.GNN_a_conv1.reset_parameters()
        self.GNN_b_conv1.reset_parameters()
        self.fc1_a.reset_parameters()
        self.fc1_b.reset_parameters()

    def _permute_operation(self, data, h_b, h_a, permute=True):
        if permute:
            self.indices = np.random.permutation(h_b.shape[0])

        indices = self.indices
        h_b_swap = h_b[indices]
        label_swap = data.y[indices]
        data.aug_y = label_swap

        data.aug_train_mask = data.train_mask[indices]
        data.aug_val_mask = data.val_mask[indices]
        data.aug_test_mask = data.test_mask[indices]

        data.aug_train_anm = torch.clone(data.aug_train_mask).detach()
        data.aug_train_norm = torch.clone(data.aug_train_mask).detach()

        temp = data.aug_y == 1
        temp1 = data.aug_train_mask == True
        data.aug_train_anm = torch.logical_and(temp, temp1)

        temp = data.aug_y == 0
        temp1 = data.aug_train_mask == True
        data.aug_train_norm = torch.logical_and(temp, temp1)

        h_aug_back_a = torch.cat((h_a, h_b_swap.detach()), dim=1)
        h_aug_back_b = torch.cat((h_a.detach(), h_b_swap), dim=1)

        return h_aug_back_a, h_aug_back_b, data


class GeneralizedCELoss1(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss1, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = torch.mean(F.cross_entropy(logits, targets, reduction='none') * loss_weight)
        return loss
