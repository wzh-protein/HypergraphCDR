import numpy as np
import pandas as pd
import pubchempy as pcp
import torch
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem


out_path = "./result/"


def get_smiles_from_pubchem(name_list):

    drug_cid_smile = []
    for name in name_list:
        try:
            compound = pcp.get_compounds(name, 'name')[0]
            smiles = compound.canonical_smiles
            pubchem_id = compound.cid
            drug_cid_smile.append([name.lower(), pubchem_id, smiles])
            print(f"获得药物 {name}\t的smiles")
        except Exception as e:
            print(f"药物 {name} 的smiles不存在 - {str(e)}")

    drug_cid_smile = pd.DataFrame(drug_cid_smile, columns=["name", "cid", "smiles"])

    drug_cid_smile.to_csv("./data/drug/drug_cid_smile.csv", sep=",")


def main():

    ic50 = pd.read_excel(
        "./data/ic50/GDSC1_fitted_dose_response.xlsx",
        engine='openpyxl',
        usecols=['DRUG_NAME', 'CELL_LINE_NAME', 'TCGA_DESC', 'LN_IC50']
    )

    upper_list = list(set(ic50['DRUG_NAME']))
    lower_upper_dict = dict(zip([string.lower() for string in upper_list], upper_list))

    ic50['DRUG_NAME'] = ic50['DRUG_NAME'].apply(lambda x: x.lower())
    ic50['CELL_LINE_NAME'] = ic50['CELL_LINE_NAME'].str.replace("[^A-Za-z0-9]", "", regex=True)

    drug_name = set(ic50["DRUG_NAME"])
    drug_desc_name = set(
        pd.read_csv("./data/drug/gdsc_250_drug_descriptors.csv", index_col=0).index
    )
    drug_name = list(drug_desc_name.intersection(drug_name))

    lower_upper_dict = {key: value for key, value in lower_upper_dict.items() if key in drug_name}
    # get_smiles_from_pubchem(lower_upper_dict.values())

    ic50 = ic50[ic50['DRUG_NAME'].isin(drug_name)]

    cell_name = pd.read_csv(
        "./data/cell/sample_info.csv",
        sep=",",
        usecols=["stripped_cell_line_name", "CCLE_Name", "DepMap_ID"]
    )
    cell_name = cell_name.dropna(subset=["stripped_cell_line_name", "CCLE_Name"])

    cell_demap_dict = dict(zip(cell_name["stripped_cell_line_name"], cell_name["DepMap_ID"]))
    ic50['CELL_LINE_NAME'] = ic50['CELL_LINE_NAME'].map(cell_demap_dict)
    ic50 = ic50.rename(columns={'CELL_LINE_NAME': 'DEPMAP_ID'})
    ic50 = ic50.dropna(subset=["DRUG_NAME", "DEPMAP_ID"])

    cell_name = cell_name[cell_name["DepMap_ID"].isin(set(ic50["DEPMAP_ID"]))]
    torch.save(cell_name, "./result/cell/cell_name.pt")

    torch.save(ic50, out_path + "ic50/drug_cell_label.pt")


if __name__ == "__main__":
    main()
