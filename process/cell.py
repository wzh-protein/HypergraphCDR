import numpy as np
import pandas as pd
import torch
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler

in_path = r"./data/cell/"
out_path = r"./result/cell/"


def main():
    copy = pd.read_csv(in_path + "CCLE_copynumber_byGene.txt", sep="\t").T
    expre = pd.read_csv(in_path + "OmicsExpressionProteinCodingGenesTPMLogp1.csv", sep=",", index_col=0)
    miRNA = pd.read_csv(in_path + "CCLE_miRNA.gct", sep="\t", skiprows=2).T
    methy = pd.read_csv(in_path + "CCLE_RRBS_tss_CpG_clusters.txt", sep="\t", low_memory=False).T

    cell_name = torch.load(out_path + "cell_name.pt")

    ccle_demap_dict = dict(zip(cell_name["CCLE_Name"], cell_name["DepMap_ID"]))

    copy = copy.iloc[5:]
    miRNA = miRNA.iloc[2:]
    methy = methy.iloc[3:]

    copy = copy.rename(index=ccle_demap_dict)
    miRNA = miRNA.rename(index=ccle_demap_dict)
    methy = methy.rename(index=ccle_demap_dict)

    copy = copy.loc[copy.index.isin(ccle_demap_dict.values()), :]
    miRNA = miRNA.loc[miRNA.index.isin(ccle_demap_dict.values()), :]
    methy = methy.loc[methy.index.isin(ccle_demap_dict.values()), :]
    expre = expre.loc[expre.index.isin(ccle_demap_dict.values()), :]

    copy = copy.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)
    miRNA = miRNA.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)
    methy = methy.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)
    expre = expre.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)

    copy.replace(['NA', 'NaN'], np.nan, inplace=True)
    miRNA.replace(['NA', 'NaN'], np.nan, inplace=True)
    methy.replace(['NA', 'NaN'], np.nan, inplace=True)
    expre.replace(['NA', 'NaN'], np.nan, inplace=True)

    copy = copy.dropna(axis=1)
    miRNA = miRNA.dropna(axis=1)
    methy = methy.dropna(axis=1)
    expre = expre.dropna(axis=1)

    copy_cell = copy.index
    miRNA_cell = miRNA.index
    methy_cell = methy.index
    expre_cell = expre.index
    public_cell = list(copy_cell.intersection(miRNA_cell).intersection(methy_cell).intersection(expre_cell))
    cell_int = dict(zip(public_cell, range(len(public_cell))))
    torch.save(cell_int, out_path + "cell_int.pt")

    copy = copy.loc[public_cell].values
    miRNA = miRNA.loc[public_cell].values
    methy = methy.loc[public_cell].values
    expre = expre.loc[public_cell].values

    min_max = MinMaxScaler()
    copy = torch.tensor(min_max.fit_transform(copy)).float()
    miRNA = torch.tensor(min_max.fit_transform(miRNA)).float()
    methy = torch.tensor(min_max.fit_transform(methy)).float()
    expre = torch.tensor(min_max.fit_transform(expre)).float()

    cell_feat = torch.hstack((copy, miRNA, methy, expre))

    torch.save(copy, out_path + 'copy.pt')
    torch.save(miRNA, out_path + 'miRNA.pt')
    torch.save(methy, out_path + 'methy.pt')
    torch.save(expre, out_path + 'expre.pt')
    torch.save(cell_feat, out_path + 'cell_feat.pt')
    torch.save(public_cell, out_path + 'public_cell.pt')


if __name__ == "__main__":
    main()
