import os
import random
import numpy as np
from torch import nn as nn
from config import config
from model.model import Auto_Encoder
import torch
from sklearn.preprocessing import MinMaxScaler
import datetime
import torch.utils.data as Data


args = config.parse()
torch.manual_seed(args.seed)
np.random.seed(args.seed)  

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

lr = 0.0001

def train_ae(model, train_dataload, test_dataload):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.pat)

    loss_func = nn.MSELoss()
    best_model = model
    best_loss = 100

    for epoch in range(1, 5000 + 1):
        start = datetime.datetime.now()
        model.train()
        for train_feat in train_dataload:
            y = train_feat
            encoded, decoded = model(train_feat)
            train_loss = loss_func(decoded, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        model.eval()
        for test_feat in test_dataload:
            te_y = test_feat
            encoded, decoded = model(test_feat)
            test_loss = loss_func(decoded, te_y)
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                best_model = model
            schedular.step(test_loss)
        end = datetime.datetime.now()
        print(epoch, 'tr_loss = ', train_loss.item(), "te_loss:", test_loss.item(),
              "best_loss", best_loss, "time:", (end - start).seconds)
    return best_model 


def main():
    cell_feat = torch.load("./result/cell/cell_feat.pt")

    cell_feat_dim = cell_feat.shape[-1]
    model = Auto_Encoder(cell_feat_dim, 196).to(device)
    train_list = random.sample(cell_feat.tolist(), int(0.9 * len(cell_feat)))
    test_list = [item for item in cell_feat.tolist() if item not in train_list]
    train = torch.tensor(train_list).float().to(device)
    test = torch.tensor(test_list).float().to(device)
    train_dataload = Data.DataLoader(train, batch_size=len(train_list), shuffle=True)
    test_dataload = Data.DataLoader(test, batch_size=len(test_list), shuffle=True)
    best_model = train_ae(model, train_dataload, test_dataload)
    torch.save(best_model.output(cell_feat.to(device)), "./result/cell/cell_feat_ae.pt")

if __name__ == '__main__':
    main()
