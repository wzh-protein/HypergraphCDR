import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch_sparse
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from sklearn.metrics import *
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


def data_split(X, Y):
    num = len(X)
    train_num = int(num * 0.8)
    train = random.sample(range(num), train_num)
    train_X = X[train]
    train_Y = Y[train]
    test = [elem for elem in range(num) if elem not in train]
    test_X = X[test]
    test_Y = Y[test]
    return train_X, train_Y, test_X, test_Y


def data_process(X, Y):
    sen_X = []
    res_X = []
    for index, label in enumerate(Y):
        if label == 1:
            sen_X.append(X[index].tolist())
        else:
            res_X.append(X[index].tolist())

    sen_X = torch.tensor(sen_X)
    sen_Y = torch.zeros(len(sen_X))
    res_X = torch.tensor(res_X)
    res_Y = torch.zeros(len(res_X))

    return sen_X, sen_Y, res_X, res_Y


def drug_cell_feat(drug_feat, cell_feat, drug_cell_pair):
    X = []
    Y = []
    for i, row in enumerate(drug_cell_pair):
        cell_demap = row[0]
        drug_cid = row[2]
        ic50 = row[3]
        if drug_cid in cid_int.keys() and cell_demap in demap_int.keys():
            drug_feature = drug_feat[cid_int[drug_cid]]
            cell_feature = cell_feat[demap_int[cell_demap]]
            combined_features = torch.cat((drug_feature, cell_feature), dim=0)
            X.append(combined_features.tolist())
            Y.append(ic50)
    return torch.tensor(X), torch.tensor(Y).long()


def metrics_graph(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    pcc = pearsonr(y_true, y_pred)[0]
    scc = spearmanr(y_true, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return pcc, scc, r2, rmse


def cos_sim(X):
    norm_X = X / X.norm(dim=1).unsqueeze(1)
    X_sim = torch.mm(norm_X, norm_X.t())
    return X_sim


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cal_degree_of_each_pair(nlist, G):
    degree_matrix = torch.zeros(len(nlist), len(nlist))
    for hyperedge in G.keys():
        e = G[hyperedge]
        for node_i in e:
            for node_j in e:
                degree_matrix[node_i, node_j] = degree_matrix[node_i, node_j] + 1
    return degree_matrix


def aug_node(overlappness, cut_off, p):
    weights = overlappness.clone()

    weights = (weights.max() - weights) / (weights.max() - weights.mean())
    if p < 0. or p > 1.:
        raise ValueError(
            'Dropout probability has to be between 0 and 1, '
            'but got {}'.format(p)
        )

    weights = weights * p
    weights = weights.where(
        weights < cut_off,
        torch.ones_like(weights) * cut_off
    )

    sel_mask = ~torch.bernoulli(1. - weights).to(torch.bool)
    return sel_mask


def cal_overlappness(nlist, G):
    edgedict = dict()
    for i in nlist:
        edgedict[i] = []

    for hyperedge in G.keys():
        e = G[hyperedge]
        for node in e:
            edgedict[node] += e

    overlappness = []
    for E in edgedict.keys():
        subgraph = edgedict[E]
        subgraph_set = set(subgraph)
        up = len(subgraph)
        down = len(subgraph_set)

        if up != 0 and down != 0:
            overlappness.append(up / down)
        else:
            overlappness.append(0)

    overlappness = torch.Tensor(overlappness)
    return overlappness


def normalize_l2(X):
    rownorm = X.detach().sum(dim=1, keepdims=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X


def cal_homogeneity_hyperedge(nlist, G):
    degree_matrix = cal_degree_of_each_pair(nlist, G)
    homogeneity = []
    for hyperedge in G.keys():
        e = G[hyperedge]
        if len(e) > 1:
            homo = 0
            for node_i in e:
                for node_j in e:
                    if node_i != node_j:
                        homo = homo + degree_matrix[node_i, node_j].item()

            homo = homo / (len(e) * (len(e) - 1))
            homogeneity.append(sigmoid(homo))
        else:
            homogeneity.append(1)

    homogeneity = torch.Tensor(homogeneity)
    return homogeneity


def my_loss(pos_y, neg_y, y, args):
    regression_loss = F.mse_loss(pos_y[:, 1], y)

    pos_y = pos_y.T
    neg_y = neg_y.T.flip(0)

    l1 = cal_loss(pos_y, neg_y, args)
    l2 = cal_loss(neg_y, pos_y, args)

    contrastive_loss = (l1 + l2) * 0.5
    contrastive_loss = contrastive_loss.mean()

    return regression_loss + contrastive_loss


def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def cal_loss(z1, z2, args):
    f = lambda x: torch.exp(x / args.tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag() /
        (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    )


def initialise(G, drug_dim, cell_dim):
    hyperedge_index = [[], []]
    for e, vs in G.items():
        for v in vs:
            hyperedge_index[0].append(v)
            hyperedge_index[1].append(e)
    return hyperedge_index


def hypergraph_feat_extr(drug_feature, cell_feature):
    drug_feature = drug_feature.numpy()
    cell_feature = cell_feature.detach().numpy()

    m = drug_feature.shape[0]
    n = cell_feature.shape[0]
    corr_matrix = np.empty((m, n))

    for i in range(m):
        row1 = drug_feature[i, :]
        for j in range(n):
            row2 = cell_feature[j, :]
            corr_matrix[i, j] = np.corrcoef(row1, row2)[0, 1]

    corr_matrix_sort = np.argsort(corr_matrix, axis=1)
    corr_pos_index = corr_matrix_sort[:, -10:]
    corr_neg_index = corr_matrix_sort[:, :10]
    pos_row = corr_pos_index.reshape(-1)
    neg_row = corr_neg_index.reshape(-1)
    col = np.repeat(np.arange(m), 10)
    hyperedge_pos_index = np.vstack((pos_row, col))
    hyperedge_neg_index = np.vstack((neg_row, col))
    return hyperedge_pos_index, hyperedge_neg_index


class Drug_Dataset(Dataset):
    def __init__(self, drug_data):
        super(Drug_Dataset, self).__init__()
        self.drug_data = drug_data

    def __len__(self):
        return len(self.drug_data)

    def __getitem__(self, index):
        return self.drug_data[index]


def custom_collate(batch):
    return Batch.from_data_list(batch)


class MyDataset(Dataset):
    def __init__(self, y):
        super(MyDataset, self).__init__()
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.y[index]
