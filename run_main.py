import datetime
import os
import sys
import torch
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import pandas as pd

from model.model import *
from model.utils import *
from config import config
from sklearn.model_selection import KFold

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import warnings
warnings.filterwarnings("ignore")

args = config.parse()

# ===================== Seed =====================
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


# =========================================================
#                     Metrics
# =========================================================
def metrics_graph(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    pcc = pearsonr(y_true, y_pred)[0]
    scc = spearmanr(y_true, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return pcc, scc, r2, rmse


# =========================================================
#                     Logger
# =========================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# =========================================================
#               Training Function
# =========================================================
def training(model, drug_feature, pos_idx, neg_idx, cell_feature, train_dl, test_dl, ckpt_path):

    ckpt_dir = os.path.dirname(ckpt_path)
    if ckpt_dir != "" and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.pat)

    best_r2 = -1
    best_pcc = -1
    best_scc = -1
    best_rmse = float("inf")
    best_loss = float("inf")
    best_epoch = -1
    best_metrics = None

    for epoch in range(1, args.epoch + 1):

        # ===================== Train =====================
        model.train()
        train_losses = []
        for batch in train_dl:
            pos_y, neg_y = model(drug_feature, pos_idx, neg_idx, cell_feature, batch)
            y = batch[:, 2].float().to(device)

            loss = my_loss(pos_y, neg_y, y, args)

            pcc, scc, r2, rmse = metrics_graph(
                y.detach().cpu().numpy(),
                pos_y[:, 1].detach().cpu().numpy()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # ===================== Test =====================
        model.eval()
        test_losses = []
        preds, labels = [], []

        with torch.no_grad():
            for batch in test_dl:
                te_pos_y, te_neg_y = model(drug_feature, pos_idx, neg_idx, cell_feature, batch)
                te_y = batch[:, 2].float().to(device)

                te_loss = my_loss(te_pos_y, te_neg_y, te_y, args)

                te_pcc, te_scc, te_r2, te_rmse = metrics_graph(
                    te_y.detach().cpu().numpy(),
                    te_pos_y[:, 1].detach().cpu().numpy()
                )

                schedular.step(te_loss)

                # 只保存最优 test_loss 对应的模型
                if te_loss.item() < best_loss:
                    best_loss = te_loss.item()

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "test_loss": best_loss,
                            "metrics": {
                                "pcc": te_pcc,   
                                "scc": te_scc,   
                                "r2": te_r2,     
                                "rmse": te_rmse 
                            }
                        },
                        ckpt_path
                    )

                best_r2 = max(te_r2, best_r2)
                best_pcc = max(te_pcc, best_pcc)
                best_scc = max(te_scc, best_scc)
                best_rmse = min(te_rmse, best_rmse) 


        print(f"[Epoch {epoch}] Train | Loss={train_loss:.4f} "
              f"PCC={pcc:.4f} SCC={scc:.4f} R2={r2:.4f} RMSE={rmse:.4f}")

        print(f"[Epoch {epoch}] Test  | Loss={te_loss:.4f} "
              f"PCC={te_pcc:.4f} SCC={te_scc:.4f} "
              f"R2={te_r2:.4f} RMSE={te_rmse:.4f} "
              f"Best_R2={best_r2:.4f} Best_RMSE={best_rmse:.4f}")

    return best_epoch, best_loss, best_r2, best_pcc, best_scc, best_rmse


# =========================================================
#                        Main
# =========================================================
def main():

    # ===================== log =====================
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"./training_log_{now}.txt"
    sys.stdout = Logger(log_path)
    print(f"Logging to {log_path}\n")

    # ===================== Load features =====================
    drug_feature = torch.tensor(torch.load("./result/drug/drug_feat.pt")).float().to(device)
    cell_feature = torch.load("./result/cell/cell_feat_ae.pt", map_location="cpu").to(device)

    # ===================== Drug-Cell Pair =====================
    drug_cell_pair = torch.load("./result/ic50/drug_cell_label.pt")
    drug_cell_pair = pd.DataFrame(drug_cell_pair, columns=["DEPMAP_ID", "TCGA_DESC", "DRUG_NAME", "LN_IC50"])
    drug_cell_pair = drug_cell_pair.drop("TCGA_DESC", axis=1)
    drug_cell_pair = drug_cell_pair.drop_duplicates(subset=['DEPMAP_ID', 'DRUG_NAME'])

    drug_int = torch.load("./result/drug/drug_int.pt")
    cell_int = torch.load("./result/cell/cell_int.pt")

    drug_cell_pair["DRUG_ID"] = drug_cell_pair["DRUG_NAME"].map(drug_int)
    drug_cell_pair["CELL_ID"] = drug_cell_pair["DEPMAP_ID"].map(cell_int)
    drug_cell_pair = drug_cell_pair.dropna(subset=["DRUG_ID", "CELL_ID"])

    pair_arr = drug_cell_pair[["CELL_ID", "DRUG_ID", "LN_IC50"]].values
    pair_tensor = torch.tensor(pair_arr, dtype=torch.float32).to(device)

    drug_num = drug_feature.shape[0]
    cell_num = cell_feature.shape[0]

    # ===================== Hypergraph =====================
    pos_idx, neg_idx = hypergraph_feat_extr(drug_feature.cpu(), cell_feature.cpu())
    pos_idx = torch.tensor(pos_idx).long().to(device)
    neg_idx = torch.tensor(neg_idx).long().to(device)

    # ===================== 5-Fold CV =====================
    kf = KFold(n_splits=5, shuffle=True, random_state=88)
    results = []

    fold = 1
    for train_idx, test_idx in kf.split(pair_tensor):
        print(f"\n====================== Fold {fold} ======================")

        model = HyperCDR(
            drug_num, cell_num,
            drug_feature.shape[-1], cell_feature.shape[-1],
            args, device
        ).to(device)

        train_dl = Data.DataLoader(MyDataset(pair_tensor[train_idx]), batch_size=512, shuffle=True)
        test_dl = Data.DataLoader(MyDataset(pair_tensor[test_idx]), batch_size=len(test_idx), shuffle=True)

        best_epoch, best_loss, best_r2, best_pcc, best_scc, best_rmse = training(
            model, drug_feature, pos_idx, neg_idx, cell_feature, train_dl, test_dl, ckpt_path=f"./ckpt/fold{fold}_best_model.pt"
        )

        results.append({
            "fold": fold,
            "loss": best_loss,
            "pcc": best_pcc,
            "scc": best_scc,
            "r2": best_r2,
            "rmse": best_rmse
        })

        fold += 1

    # ===================== 5-Fold Summary =====================
    print("\n====================== 5-Fold Summary ======================")
    for r in results:
        print(f"Fold {r['fold']}: Loss={r['loss']:.4f}, PCC={r['pcc']:.4f}, "
            f"SCC={r['scc']:.4f}, R2={r['r2']:.4f}, RMSE={r['rmse']:.4f}")

    mean_loss = np.mean([r["loss"] for r in results])
    std_loss = np.std([r["loss"] for r in results])

    mean_pcc = np.mean([r["pcc"] for r in results])
    std_pcc = np.std([r["pcc"] for r in results])

    mean_scc = np.mean([r["scc"] for r in results])
    std_scc = np.std([r["scc"] for r in results])

    mean_r2 = np.mean([r["r2"] for r in results])
    std_r2 = np.std([r["r2"] for r in results])

    mean_rmse = np.mean([r["rmse"] for r in results])
    std_rmse = np.std([r["rmse"] for r in results])

    print("\n====================== Mean Results ======================")
    print(f"Mean Loss = {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Mean PCC  = {mean_pcc:.4f} ± {std_pcc:.4f}")
    print(f"Mean SCC  = {mean_scc:.4f} ± {std_scc:.4f}")
    print(f"Mean R2   = {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Mean RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")

    print(f"\nTraining log saved to: {log_path}")



if __name__ == "__main__":
    main()