# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         get_best_model
# Description:  test in train dataset
# Author:       梁超
# Date:         2021/10/21
# -------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from res_model import *
from colorama import Fore
from metric import *
from res_train import SeedDataset


def test_in_train(dataloader, model, loss_fn, device):
    model.eval()
    model = model.to(device)
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
        print(f"{Fore.CYAN + '[valid]===>'} " \
              f"loss: {loss}  acc: {100 * Accuracy(pred, Y)}%  precision: {Precision(pred, Y)}  recall: {Recall(pred, Y)}   fscore: {Fscore(pred, Y)}" \
              f"{'' + Fore.RESET}")


if __name__ == '__main__':

    torch.manual_seed(777)
    device = torch.device('cuda')

    epochs = 300

    model = ResNet(ResidualBlock, [2, 2, 2])
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    valid_dataset = SeedDataset("./data/v1/train.csv")
    valid_dataloader = DataLoader(valid_dataset, batch_size=100, shuffle=False)

    for t in range(200, epochs):
        path = "./checkpoints/"+str(t)+"_epoc.pt"
        print(path)
        model.load_state_dict(torch.load(path))
        test_in_train(valid_dataloader, model, loss_fn, device)
