import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from colorama import Fore
import pandas as pd
import os
import argparse

from models.metric import *
from models.baseline.baseline_model import *
from models.resnet.res_model import *
from models.LCNet.LCNet_model import *
from preprocess import *


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
    for _, (X, y) in enumerate(dataloader):
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


def valid(dataloader, model, loss_fn, device):
    model.eval()

    num_dataset = len(dataloader.dataset)
    loss = 0

    with torch.no_grad():
        pred, Y = [], []
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            logit = model(X)
            loss += loss_fn(logit, y).item()

            pred.append(logit.argmax(1))
            Y.append(y)

        loss /= num_dataset

        pred = torch.cat(pred)
        Y = torch.cat(Y)

        metric = {'acc': 0, 'precision': 0, 'recall': 0, 'Fscore': 0}
        metric['acc'] = Accuracy(pred, Y)
        metric['precision'] = Precision(pred, Y)
        metric['recall'] = Recall(pred, Y)
        metric['Fscore'] = Fscore(pred, Y)

        print(f"{Fore.CYAN + '[valid]===>'} "
              f"loss: {loss}  acc: {metric['acc']}  precision: {metric['precision']}  recall: {metric['recall']}   fscore: {metric['Fscore']}"
              f"{'' + Fore.RESET}")

        return loss


def data_prep():
    print("----------Process the dataset----------")
    print("merging the user_base_info and user_his_features ...")
    merge_base_feature()
    print("Done!\nadding user_track information into all_info ...")
    add_track()
    print("Done!\ngenerating the train_data and valid_data ...")
    generate_train_valid()
    print("Done!\ngenerating the balanced valid ...")
    valid_balanced("data/28_dimension/valid.csv")
    valid_balanced("data/33_dimension/valid.csv")
    print("Done!\ngenerating the normalized dataset ...")
    normalize()
    print("Done!\ngenerating the ML dataset ...")
    generate_ML_train_valid()
    print("Done!\ngenerating the test_data ...")
    generate_test("data/28_dimension")
    generate_test("data/33_dimension")
    generate_test("data/33_normalized")
    print("Done!\nbalancing the ML dataset ...")
    ML_balanced_train("data/33_normalized/train.csv")
    ML_balanced_valid("data/33_normalized/valid.csv")
    print("Done!")
    print("-------------Process Over--------------")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    torch.manual_seed(777)

    # Data Preprocess
    data_prep()

    # Train
    for model_name in ["baseline", "resnet", "LCNet"]:

        print(f"\n{Fore.GREEN + 'Begin to train ' + model_name}:\n")

        # generate checkpoints dir
        if os.path.isdir(f"checkpoints") == 0:
            os.mkdir(f"checkpoints")
        if os.path.isdir(f"checkpoints/{model_name}") == 0:
            os.mkdir(f"checkpoints/{model_name}")

        # model parameters
        if model_name == "LCNet":
            path = "data/33_dimension"
            batch_size, in_features, out_features, lr, positive_weight = 24, 33, 2, 1e-4, 1.5
        else:
            path = "data/28_dimension"
            batch_size, in_features, out_features, lr, positive_weight = 30, 28, 2, 1e-3, 2.33

        device = torch.device(args.device)
        epochs = 300
        loss_fn = (nn.CrossEntropyLoss()).to(device)

        if model_name == "baseline":
            model = Fake1DAttention(in_features, out_features).to(device)
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        elif model_name == "resnet":
            model = ResNet(ResidualBlock, [2, 2, 2], in_features).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif model_name == "LCNet":
            model = CTNet(batch_size, in_features, out_features).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

        # load dataset
        train_dataset = SeedDataset(f"{path}/train.csv")
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataset = SeedDataset(f"{path}/valid.csv")
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=1, shuffle=False)

        # train
        for t in range(epochs):
            print(f"{Fore.GREEN + '===>'} Epoch {t + 1} {'' + Fore.RESET}\n"
                  "---------------------------------------")

            train(train_dataloader, model, loss_fn,
                  optimizer, device, positive_weight)
            valid(valid_dataloader, model, loss_fn, device)

            torch.save(model.state_dict(),
                       f"checkpoints/{model_name}/{t}_epoc.pt")

        print(f"\n{Fore.GREEN + model_name + ' trains over'}:\n")
