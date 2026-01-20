import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import pubchempy as pcp


def main():
    drug_feat = pd.read_csv("./data/drug/gdsc_250_drug_descriptors.csv", index_col=0)
    ic50 = torch.load("./result/ic50/drug_cell_label.pt",weights_only=False)
    drug_name = set(ic50.iloc[:, 2])

    drug_feat = drug_feat[drug_feat.index.isin(drug_name)]

    valid_cols = drug_feat.columns[~drug_feat.isna().any()]
    drug_feat = drug_feat[valid_cols]


    drug_int = dict(zip(drug_feat.index, range(len(drug_name))))
    torch.save(drug_int, "./result/drug/drug_int.pt")

    ss = StandardScaler()
    drug_feat = ss.fit_transform(drug_feat.values)

    torch.save(drug_feat, "./result/drug/drug_feat.pt")


if __name__ == "__main__":
    main()
