import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from models.baseline.baseline_model import *
from models.resnet.res_model import *
from models.LCNet.LCNet_model import *
from models.SVM.SVM import *
from models.AdaBoost.AdaBoost import *
from models.DecisionTree.DT import *
from models.RandomForest.RandomForest import *
from models.metric import P0, P1


class SeedDataset_pre(Dataset):
    def __init__(self, annotations_file):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.X = self.data.drop(columns=['id']).fillna(value=-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.X.iloc[idx].values).type(torch.FloatTensor)


class SeedDataset_test(Dataset):
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


def Net_pre(model_name, in_feature, model_path, test, outputs_path):
    if model_name == "baseline":
        model = Fake1DAttention(in_feature, 2)
    elif model_name == "resnet":
        model = ResNet(ResidualBlock, [2, 2, 2], in_feature)
    elif model_name == "LCNet":
        model = CTNet(1, in_feature, 2)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataset = SeedDataset_pre(test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    outputs = []
    for x in test_dataloader:
        logit = model(x)
        outputs.append(str(logit.argmax(1).item()))

    with open(outputs_path, 'w') as f:
        f.write('\n'.join(outputs))


def SVM_pre(clf, kernel, C, degree, train, valid, test):
    svm = SVM(clf, kernel, C, degree, train, valid, test)
    svm.fit()
    # print(
    #     f"valid:\tPrecision: {svm.P()}\tRecall: {svm.R()}\tFscore: {svm.Fscore()}")
    result = svm.predict().astype(int)

    fp = open("SVM_out.txt", "w")
    for i in range(result.shape[0]):
        fp.write(result[i].astype(str))
        fp.write('\n')

    return svm


def Ada_pre(base_estimator, n_estimators, algorithm, lr, C, train, valid, test):
    Ada = AdaBoost(base_estimator, n_estimators,
                   algorithm, lr, C, train, valid, test)
    Ada.fit()
    # print(
    #     f"valid:\tPrecision: {Ada.P()}\tRecall: {Ada.R()}\tFscore: {Ada.Fscore()}")
    result = Ada.predict().astype(int)

    fp = open("AdaBoost_out.txt", "w")
    for i in range(result.shape[0]):
        fp.write(result[i].astype(str))
        fp.write('\n')

    return Ada


def DT_pre(max_depth, criterion, splitter, train, valid, test):
    decision_tree = DT(max_depth=max_depth, criterion=criterion, splitter=splitter,
                       trainfile=train, validfile=valid, testfile=test)
    decision_tree.fit()
    # print(
    #     f"valid:\tPrecision: {decision_tree.P()}\tRecall: {decision_tree.R()}\tFscore: {decision_tree.Fscore()}")
    fp = open("DecisionTree_out.txt", "w")
    result = decision_tree.predict().astype(int)
    for i in range(result.shape[0]):
        fp.write(result[i].astype(str))
        fp.write('\n')

    return decision_tree


def RF_pre(max_depth, random_state, train, valid, test):
    forest = RandomForest(max_depth=max_depth, random_state=random_state,
                          trainfile=train, validfile=valid, testfile=test)
    forest.fit()
    # print(
    #     f"valid:\tPrecision: {forest.P()}\tRecall: {forest.R()}\tFscore: {forest.Fscore()}")
    fp = open("RandomForest_out.txt", "w")
    result = forest.predict().astype(int)
    for i in range(result.shape[0]):
        fp.write(result[i].astype(str))
        fp.write('\n')
    return forest


def valid(Net, dataloader, args_model, loss_fn, device):
    if(Net == "ResNet"):
        model = ResNet(ResidualBlock, [2, 2, 2], 28)
    elif(Net == "LCNet"):
        model = CTNet(batch=1, in_channels=33, out_channels=2)
    elif(Net == "Baseline"):
        model = Fake1DAttention(28, 2)

    model.load_state_dict(torch.load(args_model))
    model.eval()

    model = model.to(device)
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

        return round(P1(pred, Y), 4), round(P0(pred, Y), 4)


def Net_test(model_name, device, valid_balanced, model_path):
    valid_data = DataLoader(SeedDataset_test(valid_balanced),
                            batch_size=1, shuffle=False)
    if model_name == 'baseline':
        return valid("Baseline", valid_data, model_path, nn.CrossEntropyLoss().to(device), device)
    elif model_name == 'ResNet':
        return valid("ResNet", valid_data, model_path, nn.CrossEntropyLoss().to(device), device)
    elif model_name == 'LCNet':
        return valid("LCNet", valid_data, model_path, nn.CrossEntropyLoss().to(device), device)


def SVM_test(clf, kernel, C, degree, train, valid, test):
    svm = SVM(clf, kernel, C, degree, train, valid, test)
    svm.fit()
    return svm.P1(), svm.P0()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # select the best epoc for Nets
    print("selecting the best epoc for ResNet and LCNet ...")
    res_best_epoc = 24
    res_best_epoc = 273
    LC_best_epoc = 131
    r1, r0, c1, c0, b1, b0 = 0, 0, 0, 0, 0, 0
    for i in range(300):
        # baseline
        B_P1, B_P0 = Net_test(
            "baseline", device, "data/28_dimension/valid_balanced.csv", f"checkpoints/baseline/{i}_epoc.pt")

        # ResNet
        R_P1, R_P0 = Net_test(
            "ResNet", device, "data/28_dimension/valid_balanced.csv", f"checkpoints/resnet/{i}_epoc.pt")
        # LCNet
        L_P1, L_P0 = Net_test(
            "LCNet", device, "data/33_dimension/valid_balanced.csv", f"checkpoints/LCNet/{i}_epoc.pt")
        if B_P1+B_P0 > b1+b0:
            base_best_epoc = i
            b1 = B_P1
            b0 = B_P0
        if R_P1 + R_P0 > r1+r0:
            res_best_epoc = i
            r1 = R_P1
            r0 = R_P0
        if L_P1 + L_P0 > c1+c0:
            LC_best_epoc = i
            c1 = L_P1
            c0 = L_P0
    print(
        f"Done!\nbaseline: {base_best_epoc}\tResNet: {res_best_epoc}\tLCNet: {LC_best_epoc}\n")

    # Part1: Predict
    # baseline
    Net_pre("baseline", 28, f"checkpoints/baseline/{base_best_epoc}_epoc.pt",
            "data/28_dimension/test.csv", "base_out.txt")

    # resnet
    Net_pre("resnet", 28, f"checkpoints/resnet/{res_best_epoc}_epoc.pt",
            "data/28_dimension/test.csv", "res_out.txt")

    # LCNet
    Net_pre("LCNet", 33, f"checkpoints/LCNet/{LC_best_epoc}_epoc.pt",
            "data/33_dimension/test.csv", "LC_out.txt")

    # SVM
    svm = SVM_pre("SVC", "rbf", 0.6, 2, "data/33_normalized/train.csv",
                  "data/33_normalized/valid.csv", "data/33_normalized/test.csv")

    # AdaBoost
    Ada = Ada_pre("DicisionTree", 10, "SAMME.R", 1.0, 0.6, "data/33_normalized/train.csv",
                  "data/33_normalized/valid.csv", "data/33_normalized/test.csv")

    # DecisionTree
    DT = DT_pre(3, 'gini', 'best', "data/33_normalized/train.csv",
                "data/33_normalized/valid.csv", "data/33_normalized/test.csv")

    # Random Forest
    RF = RF_pre(5, 0, "data/33_normalized/train.csv",
                "data/33_normalized/valid.csv", "data/33_normalized/test.csv")

    # Part2: Vote
    # Baseline
    Base_P1, Base_P0 = Net_test("baseline", device, "data/28_dimension/valid_balanced.csv",
                                f"checkpoints/baseline/{base_best_epoc}_epoc.pt")

    # ResNet
    Res_P1, Res_P0 = Net_test("ResNet", device, "data/28_dimension/valid_balanced.csv",
                              f"checkpoints/resnet/{res_best_epoc}_epoc.pt")

    # LCNet
    LC_P1, LC_P0 = Net_test("LCNet", device, "data/33_dimension/valid_balanced.csv",
                            f"checkpoints/LCNet/{LC_best_epoc}_epoc.pt")

    # SVM
    SVM_P1, SVM_P0 = svm.P1(), svm.P0()

    # AdaBoost
    Ada_P1, Ada_P0 = Ada.P1(), Ada.P0()

    # Dicision Tree
    DT_P1, DT_P0 = DT.P1(), DT.P0()

    # Random Forest
    RF_P1, RF_P0 = RF.P1(), RF.P0()

    # print(
    #     f"Base_P1: {Base__P1}\tBase_P0: {Base_P0}\nRes_P1: {Res_P1}\tRes_P0: {Res_P0}\nLC_P1: {LC_P1}\tLC_P0: {LC_P0}\nSVM_P1: {SVM_P1}\tSVM_P0: {SVM_P0}\nAda_P1: {Ada_P1}\tAda_P0: {Ada_P0}\nDT_P1: {DT_P1}\tDT_P0: {DT_P0}\n")
    print(
        f"Base_P1: {Base_P1}\tBase_P0: {Base_P0}\nRes_P1: {Res_P1}\tRes_P0: {Res_P0}\nLC_P1: {LC_P1}\tLC_P0: {LC_P0}\nSVM_P1: {SVM_P1}\tSVM_P0: {SVM_P0}\nAda_P1: {Ada_P1}\tAda_P0: {Ada_P0}\nDT_P1: {DT_P1}\tDT_P0: {DT_P0}\nRF_P1: {RF_P1}\tRF_P0: {RF_P0}")

    result = open("output.txt", "w")
    P1_sum = Res_P1 + LC_P1 + SVM_P1 + DT_P1 + Ada_P1 + Base_P1 + RF_P1
    P0_sum = Res_P0 + LC_P0 + SVM_P0 + DT_P0 + Ada_P0 + Base_P0 + RF_P0
    with open("base_out.txt") as Base_r, open("res_out.txt") as Res_r, open("LC_out.txt") as LC_r, open("RandomForest_out.txt") as RF_r, open("SVM_out.txt") as SVM_r, open("DecisionTree_out.txt") as DT_r, open("AdaBoost_out.txt") as Ada_r:
        sample_num = len(Res_r.readlines())
        Res_r.seek(0, 0)
        for _ in range(sample_num):
            l1 = int(Res_r.readline())
            l2 = int(LC_r.readline())
            l3 = int(RF_r.readline())
            l4 = int(SVM_r.readline())
            l5 = int(DT_r.readline())
            l6 = int(Ada_r.readline())
            l7 = int(Base_r.readline())

            r1 = (l1*Res_P1 + l2*LC_P1 + l3*RF_P1 + l4 *
                  SVM_P1 + l5*DT_P1 + l6*Ada_P1 + l7*Base_P1) / P1_sum
            r0 = ((1-l1)*Res_P0 + (1-l2)*LC_P0 + (1-l3)*RF_P0 + (1-l4)
                  * SVM_P0 + (1-l5)*DT_P0 + (1-l6)*Ada_P0 + (1-l7)*Base_P0) / P0_sum
            # r1 = l2*LC_P1
            # r0 = (1-l2)*LC_P0
            result.write(f"{int(r1>r0)}\n")

    os.remove("base_out.txt")
    os.remove("res_out.txt")
    os.remove("LC_out.txt")
    os.remove("SVM_out.txt")
    os.remove("DecisionTree_out.txt")
    os.remove("AdaBoost_out.txt")
    os.remove("RandomForest_out.txt")


if __name__ == "__main__":
    main()
