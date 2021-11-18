# -------------------------------------------------------------------------------
# Name:         output_vote
# Description:
# Author:       李瑞堃
# Date:         2021/10/31
# -------------------------------------------------------------------------------


from torch.utils.data import Dataset, DataLoader
from models.resnet.res_model import *
from models.LCNet.LCNet_model import *
from models.baseline.baseline_model import *
from models.Decision_tree.decision_tree import *
from models.SVM.SVM import *
from models.Adaboost.AdaBoost import *


rate = "0.5"  # 默认为6：4的正负样本比例，若要改为1：1则取rate=“0.5”


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


class SVM(SVM):
    def P1(self):
        index_ = self.classifier.predict(self.valid_x) == 1
        TP = (self.valid_y[index_] == 1).sum()
        if(index_.sum() == 0):
            return 0

        return TP / index_.sum()

    def P0(self):
        index_ = self.classifier.predict(self.valid_x) == 0
        TP = (self.valid_y[index_] == 0).sum()
        if(index_.sum() == 0):
            return 0

        return TP / index_.sum()


class AdaBoost(AdaBoost):
    def P1(self):
        index_ = self.classifier.predict(self.valid_x) == 1
        TP = (self.valid_y[index_] == 1).sum()
        if(index_.sum() == 0):
            return 0

        return TP / index_.sum()

    def P0(self):
        index_ = self.classifier.predict(self.valid_x) == 0
        TP = (self.valid_y[index_] == 0).sum()
        if(index_.sum() == 0):
            return 0

        return TP / index_.sum()


def P1(pred: torch.Tensor, y: torch.Tensor):
    index_ = pred == 1
    TP = (y[index_] == 1).sum()

    return (TP / index_.sum()).item()


def P0(pred: torch.Tensor, y: torch.Tensor):
    index_ = pred == 0
    TP = (y[index_] == 0).sum()

    return (TP / index_.sum()).item()


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

        return P1(pred, Y), P0(pred, Y)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ResNet_model', type=str,
                        default="history/test_b/ResNet/28维（未归一化）_273轮_0.8196.pt")
    parser.add_argument('--ResNet_valid', type=str,
                        default=f"data/unmodified/{rate}valid_balanced.csv")
    parser.add_argument('--ResNet_result', type=str,
                        default="history/test_b/bestmodel_on_test_b/res_output_0.8364.txt")

    parser.add_argument('--LCNet_model', type=str,
                        default="history/test_b/LCNet/best/33维（未归一化）_131轮_0.8226.pt")
    parser.add_argument('--LCNet_valid', type=str,
                        default=f"data/33_dimension/{rate}valid_banlanced.csv")
    parser.add_argument('--LCNet_result', type=str,
                        default="history/test_b/bestmodel_on_test_b/LC_output_0.8573.txt")

    parser.add_argument('--Baseline_model', type=str,
                        default="history/test_b/Fake1DAttention/28维（是否归一化）_30轮_0.6581（假）.pt")
    parser.add_argument('--Baseline_valid', type=str,
                        default=f"data/unmodified/{rate}valid_balanced.csv")
    parser.add_argument('--Baseline_result', type=str,
                        default="history/test_b/bestmodel_on_test_b/Base_output_0.7454.txt")

    parser.add_argument('--DicisionTree_valid', type=str,
                        default=f"data/ML/33_dimension/{rate}valid.csv")
    parser.add_argument('--DicisionTree_pkl', type=str,
                        default="history/test_b/ML/DecisionTree_max-depth=3_0.8142.pkl")
    parser.add_argument('--DicisionTree_result', type=str,
                        default="history/test_b/bestmodel_on_test_b/decision_tree_result_0.8201.txt")

    parser.add_argument('--SVM_feature', type=int, default=33)
    parser.add_argument('--SVM_clf', type=str, default="SVC")
    parser.add_argument('--SVM_kernel', type=str, default="rbf")
    parser.add_argument('--SVM_C', type=float, default=0.6)
    parser.add_argument('--SVM_degree', type=int, default=2)
    parser.add_argument('--SVM_result', type=str,
                        default="history/test_b/bestmodel_on_test_b/33_SVC_rbf_output_0.8215.txt")

    parser.add_argument('--Ada_feature', type=int, default=33)
    parser.add_argument('--Ada_base_estimator',
                        type=str, default="DicisionTree")
    parser.add_argument('--Ada_n_estimators', type=int, default=10)
    parser.add_argument('--Ada_algorithm', type=str, default="SAMME")
    parser.add_argument('--Ada_lr', type=float, default=1.0)
    parser.add_argument('--Ada_C', type=float, default=0.6)
    parser.add_argument('--Ada_result', type=str,
                        default="history/test_b/bestmodel_on_test_b/DicisionTree_10_1.0_33_2_rbf_C-0.6_SAMME_output_0.8244.txt")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    torch.manual_seed(777)
    device = torch.device("cpu")

    # ResNet
    Res_valid = DataLoader(SeedDataset(args.ResNet_valid),
                           batch_size=1, shuffle=False)
    Res_P1, Res_P0 = valid("ResNet", Res_valid, args.ResNet_model,
                           nn.CrossEntropyLoss().to(device), device)

    # LCNet
    LCNet_valid = DataLoader(SeedDataset(args.LCNet_valid),
                             batch_size=1, shuffle=False)
    LC_P1, LC_P0 = valid("LCNet", LCNet_valid, args.LCNet_model,
                         nn.CrossEntropyLoss().to(device), device)

    # Baseline
    Baseline_valid = DataLoader(SeedDataset(args.Baseline_valid),
                                batch_size=1, shuffle=False)
    Base_P1, Base_P0 = valid("Baseline", Baseline_valid, args.Baseline_model,
                             nn.CrossEntropyLoss().to(device), device)

    # SVM
    svm = SVM(args.SVM_clf, args.SVM_kernel, args.SVM_C, args.SVM_degree, f"data/ML/{args.SVM_feature}_dimension/train.csv",
              f"data/ML/{args.SVM_feature}_dimension/{rate}valid.csv", f"data/ML/{args.SVM_feature}_dimension/test_a.csv")
    svm.fit()
    SVM_P1, SVM_P0 = svm.P1(), svm.P0()

    # AdaBoost
    Ada = AdaBoost(args.Ada_base_estimator, args.Ada_n_estimators, args.Ada_algorithm,  args.Ada_lr, args.Ada_C, f"data/ML/{args.Ada_feature}_dimension/train.csv",
                   f"data/ML/{args.Ada_feature}_dimension/{rate}valid.csv", f"data/ML/{args.Ada_feature}_dimension/test_b.csv")
    Ada.fit()
    Ada_P1, Ada_P0 = Ada.P1(), Ada.P0()

    # Dicision Tree
    with open(args.DicisionTree_pkl, 'rb') as input:
        decision_tree = pickle.load(input)
    DT_P1, DT_P0 = ValidModel(decision_tree, args.DicisionTree_valid)

    # Random Forest
    # TODO

    print(
        f"Res_P1: {Res_P1}\tRes_P0: {Res_P0}\nLC_P1: {LC_P1}\tLC_P0: {LC_P0}\nBase_P1: {Base_P1}\tBase_P0: {Base_P0}\nSVM_P1: {SVM_P1}\tSVM_P0: {SVM_P0}\nAda_P1: {Ada_P1}\tAda_P0: {Ada_P0}\nDT_P1: {DT_P1}\tDT_P0: {DT_P0}\n")

    result = open("history/test_b/weighted/vote_lrk.txt", "w")
    with open(args.ResNet_result) as Res_r, open(args.LCNet_result) as LC_r, open(args.Baseline_result) as Base_r, open(args.SVM_result) as SVM_r, open(args.DicisionTree_result) as DT_r, open(args.Ada_result) as Ada_r:
        for _ in range(456):
            l1 = int(Res_r.readline())
            l2 = int(LC_r.readline())
            l3 = int(Base_r.readline())
            l4 = int(SVM_r.readline())
            l5 = int(DT_r.readline())
            l6 = int(Ada_r.readline())

            r1 = l1*Res_P1 + l2*LC_P1 + l3*Base_P1 + l4*SVM_P1 + l5*DT_P1 + l6*Ada_P1
            r0 = (1-l1)*Res_P0 + (1-l2)*LC_P0 + (1-l3)*Base_P0 + \
                (1-l4)*SVM_P0 + (1-l5)*DT_P0 + (1-l6)*Ada_P0
            result.write(f"{int(r1>r0)}\n")
