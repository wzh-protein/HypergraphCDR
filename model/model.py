import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TransformerFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        x, gate = self.fc1(x).chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)   # Swish gate

        x = self.drop(x)
        x = self.fc2(x)
        return residual + x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, drop):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers += [
                nn.LayerNorm(dims[i]),
                Swish(),
                nn.Dropout(drop),
                nn.Linear(dims[i], dims[i + 1]),
            ]

        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class HyperConv(MessagePassing):
    def __init__(self, in_dim, out_dim, args, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.share = args.share
        self.sqrt_norm = args.sqrt_norm

        if self.share:
            self.lin = Linear(in_dim, out_dim, bias=False, weight_initializer='glorot')
            self.linV = self.lin
            self.linE = self.lin
            self.biasV = self.biasE = Parameter(torch.Tensor(out_dim))
        else:
            self.linV = Linear(in_dim, out_dim, bias=False, weight_initializer='glorot')
            self.linE = Linear(in_dim, out_dim, bias=False, weight_initializer='glorot')
            self.biasV = Parameter(torch.Tensor(out_dim))
            self.biasE = Parameter(torch.Tensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.linV.reset_parameters()
        self.linE.reset_parameters()
        zeros(self.biasV)
        zeros(self.biasE)

    def forward(self, x, hyperedge_index, hyperedge_attr, hyperedge_weight=None):
        num_nodes = x.size(0)
        num_edges = int(hyperedge_index[1].max()) + 1 if hyperedge_index.numel() > 0 else 0

        x = self.linE(x)
        hyperedge_attr = self.linV(hyperedge_attr)

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        node_idx, edge_idx = hyperedge_index
        D = scatter_add(hyperedge_weight[edge_idx], node_idx, dim=0, dim_size=num_nodes)
        B = scatter_add(x.new_ones(edge_idx.size(0)), edge_idx, dim=0, dim_size=num_edges)

        power = -0.5 if self.sqrt_norm else -1
        D = D.pow(power)
        B = B.pow(power)
        D[D == float("inf")] = 0
        B[B == float("inf")] = 0

        edge_out = self.propagate(hyperedge_index, x=x, norm=B[edge_idx], size=(num_nodes, num_edges))
        node_out = self.propagate(hyperedge_index.flip([0]), x=hyperedge_attr, norm=D[node_idx], size=(num_edges, num_nodes))

        return {
            'drug': edge_out + self.biasE,
            'cell': node_out + self.biasV
        }

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j


class Drug_Feat_Extr(nn.Module):
    def __init__(self, drug_dim, cell_dim, args, device):
        super().__init__()
        self.device = device
        self.output_dim = args.output_dim_drug
        self.layer_drug = args.layer_drug
        self.alpha = args.alpha
        self.use_ln = args.ln_drug

        self.drug_lin = nn.Linear(drug_dim, self.output_dim)
        self.cell_lin = nn.Linear(cell_dim, self.output_dim)

        self.act = Swish()
        self.ln = nn.LayerNorm(self.output_dim)

        self.hyperGCN = nn.ModuleList([
            HyperConv(self.output_dim, self.output_dim, args).to(device)
            for _ in range(self.layer_drug)
        ])

    def forward(self, drug_feat, cell_feat, hyperedge_index):
        drug_feat = self.drug_lin(drug_feat.to(self.device))
        cell_feat = self.cell_lin(cell_feat.to(self.device))
        hyperedge_index = hyperedge_index.to(self.device)

        node_feat = {'drug': drug_feat, 'cell': cell_feat}

        for i in range(self.layer_drug):
            h = self.hyperGCN[i](cell_feat, hyperedge_index, drug_feat)
            node_feat = {
                k: self.ln(self.act(v + self.alpha * node_feat[k]))
                for k, v in h.items()
            }

        return node_feat['drug']


class Cell_Feat_Extr(nn.Module):
    def __init__(self, cell_num, cell_dim, args, device):
        super().__init__()
        self.device = device
        self.output_dim = args.output_dim_cell

        self.embed = nn.Linear(cell_dim, self.output_dim)
        self.blocks = nn.ModuleList([
            TransformerFFN(self.output_dim, self.output_dim * 2, args.drop_cell)
            for _ in range(args.layer_cell)
        ])
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, cell_feat):
        x = self.embed(cell_feat.to(self.device))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class HyperCDR(nn.Module):
    def __init__(self, drug_num, cell_num, drug_dim, cell_dim, args, device):
        super().__init__()

        self.drug_feat = Drug_Feat_Extr(drug_dim, cell_dim, args, device)
        self.cell_feat = Cell_Feat_Extr(cell_num, cell_dim, args, device)

        self.proj = ProjectionHead(
            in_dim=args.output_dim_drug + args.output_dim_cell,
            hidden_dims=[args.hid_dim1, args.hid_dim2],
            out_dim=args.output_dim_all,
            drop=args.drop_all
        )

    def forward(self, drug_feature, hyperedge_pos_index, hyperedge_neg_index, cell_feature, pair):
        pos_drug = self.drug_feat(drug_feature, cell_feature, hyperedge_pos_index)
        neg_drug = self.drug_feat(drug_feature, cell_feature, hyperedge_neg_index)
        cell = self.cell_feat(cell_feature)

        # ===== 关键修复：pair 索引 =====
        pair = pair.to(pos_drug.device)
        idx_cell = pair[:, 0].long()
        idx_drug = pair[:, 1].long()

        pos_x = torch.cat((pos_drug[idx_drug], cell[idx_cell]), dim=1)
        neg_x = torch.cat((neg_drug[idx_drug], cell[idx_cell]), dim=1)

        return self.proj(pos_x), self.proj(neg_x)

    def output(self, drug_feature, hyperedge_index, cell_feature, pair):
        drug = self.drug_feat(drug_feature, cell_feature, hyperedge_index)
        cell = self.cell_feat(cell_feature)

        pair = pair.to(drug.device)
        idx_cell = pair[:, 0].long()
        idx_drug = pair[:, 1].long()

        x = torch.cat((drug[idx_drug], cell[idx_cell]), dim=1)
        return self.proj(x)[:, 1]

class Auto_Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Auto_Encoder, self).__init__()
        self.encoder = Encoder(in_dim, out_dim)
        self.decoder = Decoder(out_dim, in_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def output(self, x):
        return self.encoder(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_dim, 4 * out_dim)
        self.linear2 = nn.Linear(4 * out_dim, 2 * out_dim)
        self.linear3 = nn.Linear(2 * out_dim, out_dim)
        self.act = nn.SELU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.sig(self.linear3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(in_dim, 2 * in_dim)
        self.linear2 = nn.Linear(2 * in_dim, 4 * in_dim)
        self.linear3 = nn.Linear(4 * in_dim, out_dim)
        self.act = nn.SELU()

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x