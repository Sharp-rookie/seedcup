import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from colorama import Fore
from baseline_model import Fake1DAttention
from models.metric import *
import pandas as pd
import os
import argparse
from hyp_evol import *


rate = "0.5"  # 默认为6：4的正负样本比例，若要改为1：1则取rate=“0.5”

record = pd.DataFrame(columns=['Epoch', 'P1', 'P0', 'Fscore'])


class SeedDataset(Dataset):

    def __init__(self, annotations_file):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.data: pd.DataFrame = self.data[self.data['label'].notna()]
        self.Y = self.data['label']
        self.X = self.data.drop(columns=['id', 'label']).fillna(value=-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.X.iloc[idx].values).type(torch.FloatTensor), torch.as_tensor(self.Y.iloc[idx]).type(
            torch.LongTensor)


def train(dataloader, model, loss_fn, optimizer, device, positive_weight):
    model.train()

    Y = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        logit = model(X)
        positive_index = y == 1

        loss = loss_fn(logit, y)
        loss = (positive_weight * loss_fn(logit[positive_index], y[positive_index]) + loss_fn(logit[~positive_index], y[
            ~positive_index])) / len(X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss = loss.item()
        #     print(
        #         f"{Fore.GREEN + '[train]===>'} loss: {loss} {'' + Fore.RESET}")


def valid(dataloader, model, loss_fn, device) -> dict:
    model.eval()

    num_dataset = len(dataloader.dataset)
    loss = 0

    with torch.no_grad():
        pred, Y = [], []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            logit = model(X)
            loss += loss_fn(logit, y).item()

            pred.append(logit.argmax(1))
            Y.append(y)

        loss /= num_dataset

        pred = torch.cat(pred)
        Y = torch.cat(Y)

        # metric = {'acc': 0, 'precision': 0, 'recall': 0, 'Fscore': 0}
        # metric['acc'] = Accuracy(pred, Y)
        # metric['precision'] = Precision(pred, Y)
        # metric['recall'] = Recall(pred, Y)
        # metric['Fscore'] = Fscore(pred, Y)

        # print(f"{Fore.CYAN + '[valid]===>'} "
        #       f"loss: {loss}  acc: {metric['acc']}  precision: {metric['precision']}  recall: {metric['recall']}   fscore: {metric['Fscore']}"
        #       f"{'' + Fore.RESET}")

        return P1(pred, Y), P0(pred, Y), Fscore(pred, Y)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evol', action='store_true',
                        help="hyperparameters auto evolve")
    parser.add_argument('--train', type=str,
                        default="../../data/unmodified/train.csv")
    parser.add_argument('--valid', type=str,
                        default=f"../../data/unmodified/{rate}valid_balanced.csv")
    parser.add_argument('--device', type=str,
                        default='cpu')
    parser.add_argument('--in_feature', type=int,
                        default=28)
    # parser.add_argument('--model', help="train with last model",
    #                     type=str, default="./checkpoints/unevol/24_epoc.pt")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    torch.manual_seed(777)

    device = torch.device(args.device)

    batch_size, in_features, out_features = 30, args.in_feature, 2
    lr, positive_weight = 1e-3, 2.33
    epochs = 300

    loss_fn = (nn.CrossEntropyLoss()).to(device)

    train_dataset = SeedDataset(args.train)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = SeedDataset(args.valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    if(os.path.isdir(f"../../checkpoints") == 0):
        os.mkdir(f"../../checkpoints")
    if(os.path.isdir(f"../../checkpoints/baseline") == 0):
        os.mkdir(f"../../checkpoints/baseline")

    # Direct train
    if args.evol == False:
        print(
            f"\nepochs: {epochs}\ndevice: {device}\nin_feature: {args.in_feature}\ntrain_set: {args.train}\nvalid_set: {args.valid}\n")

        if(os.path.isdir("../../checkpoints/baseline/unevol") == 0):
            os.mkdir("../../checkpoints/baseline/unevol")

        model = Fake1DAttention(in_features, out_features).to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr)

        for t in range(epochs):
            # print(f"{Fore.GREEN + '===>'} Epoch {t + 1} {'' + Fore.RESET}\n"
            #       "---------------------------------------")
            train(train_dataloader, model, loss_fn,
                  optimizer, device, positive_weight)
            P1_, P0_, Fscore_ = valid(valid_dataloader, model, loss_fn, device)
            record.loc[t] = (str(t), str(P1_), str(P0_), str(Fscore_))

            torch.save(model.state_dict(),
                       f"../../checkpoints/baseline/unevol/{t}_epoc.pt")

        record.to_csv("record.csv")

    # Train after hyperparameter evolution
    else:
        if(os.path.isdir("../../checkpoints/baseline/evol") == 0):
            os.mkdir("../../checkpoints/baseline/evol")

        hyp = {'lr': 1e-3,
               'positive_weight': 2.33}

        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr': (1, 1e-5, 1),  # learning rate
                'positive_weight': (1, 0.5, 5)}

        # Hyperparameter evolution
        for g in range(10):
            model = Fake1DAttention(in_features, out_features).to(device)

            if(os.path.isdir(f"../../checkpoints/baseline/evol/generate_{g}") == 0):
                os.mkdir(f"../../checkpoints/baseline/evol/generate_{g}")

            # Get hyperparameter from gene bank
            lr, positive_weight = GetHyper(meta, hyp)
            optimizer = optim.Adagrad(model.parameters(), lr=lr)

            # Train
            for t in range(30):
                print(
                    "---------------------------------------\n"f"{Fore.GREEN + '===>'} Generate[{g}] --- Epoch{t + 1} {'' + Fore.RESET}:")
                train(train_dataloader, model, loss_fn,
                      optimizer, device, positive_weight)
                metric = valid(valid_dataloader, model, loss_fn, device)
                torch.save(model.state_dict(),
                           f"../../checkpoints/baseline/evol/generate_{g}/{g}_{t}_epoc.pt")

            # Update the gene bank with fitness values
            Update_gene(hyp, metric)

        # Train with best hyperparameters
        x = np.loadtxt('evolve.txt', ndmin=2)
        lr = x[0][4]
        positive_weight = x[0][5]
        print(
            f"best hyperparameter : lr={lr}    positive_weight={positive_weight}\n")

        model = Fake1DAttention(in_features, out_features).to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr)

        for t in range(epochs):
            print(f"{Fore.GREEN + '===>'} Epoch {t + 1} {'' + Fore.RESET}\n"
                  "---------------------------------------")
            train(train_dataloader, model, loss_fn,
                  optimizer, device, positive_weight)
            valid(valid_dataloader, model, loss_fn, device)
            if(os.path.isdir(f"../../checkpoints/baseline/evol/best/best_epoc_{t}.pt") == 0):
                os.mkdir(
                    f"../../checkpoints/baseline/evol/best/best_epoc_{t}.pt")
            torch.save(model.state_dict(),
                       f"../../checkpoints/baseline/evol/best/best_epoc_{t}.pt")
