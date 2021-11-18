import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import argparse


rate = "0.5"  # 默认为6：4的正负样本比例，若要改为1：1则取rate=“0.5”


class SGD:
    def __init__(self, trainfile, validfile, testfile):
        super(SGD, self).__init__()

        train: pd.DataFrame = pd.read_csv(trainfile)
        train: pd.DataFrame = train[train['label'].notna()]
        valid: pd.DataFrame = pd.read_csv(validfile)
        valid: pd.DataFrame = valid[valid['label'].notna()]
        test: pd.DataFrame = pd.read_csv(testfile)

        self.train_y = train['label']
        self.train_x = train.drop(columns=['id', 'label']).fillna(value=-1)
        self.valid_y = valid['label']
        self.valid_x = valid.drop(columns=['id', 'label']).fillna(value=-1)
        self.test = test.drop(columns=['id']).fillna(value=-1)

        self.classifier = Pipeline([
            # ('pca',PCA()),
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(
                alpha=0.0001,
                average=False,  # averaged SGD (ASGD)
                early_stopping=False,
                epsilon=0.1,
                eta0=0.0,
                fit_intercept=True,
                learning_rate='optimal',
                loss='hinge',  # loss_fnn: hinge、log、modified_huber
                max_iter=1000,
                n_iter_no_change=5,
                penalty='l2',  # 惩罚方法: l1、l2、elasticnet
                l1_ratio=0.15,  # (1 - l1_ratio) * L2 + l1_ratio * L1
                shuffle=True,  # 每次迭代后打乱训练数据
                warm_start=False))])

    def fit(self):
        self.classifier.fit(self.train_x, self.train_y)

    def P(self):
        index_ = self.classifier.predict(self.valid_x) == 1
        TP = (self.valid_y[index_] == 1).sum()
        if(index_.sum() == 0):
            return 0

        return round(TP / index_.sum(), 4)

    def R(self):
        index_ = self.valid_y == 1
        TP = (self.classifier.predict(self.valid_x)[index_] == 1).sum()
        if(index_.sum() == 0):
            return 0

        return round(TP / index_.sum(), 4)

    def Fscore(self):
        P = self.P()
        R = self.R()
        if(P + R == 0):
            return 0

        return round(5 * P * R / (3 * P + 2 * R), 4)

    def predict(self):
        return self.classifier.predict(self.test).astype(float)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature', type=int, default=33)
    parser.add_argument('--auto', action='store_true',
                        help="auto predict by different hyperparameters")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if(args.auto == False):
        print(f"SGD\nfeature: {args.feature}")

        sgd = SGD(f"../../data/ML/{args.feature}_dimension/train.csv",
                  f"../../data/ML/{args.feature}_dimension/{rate}valid.csv", f"../../data/ML/{args.feature}_dimension/test_b.csv")

        sgd.fit()
        print(
            f"valid:\tPrecision: {sgd.P()}\tRecall: {sgd.R()}\tFscore: {sgd.Fscore()}")
        result = sgd.predict().astype(int)

        # fp = open(
        #     f"{args.base_estimator}_{args.n_estimators}_{args.lr}_{args.feature}_{args.degree}_{args.kernel}_C-{args.C}_output.txt", "w")
        # for i in range(result.shape[0]):
        #     fp.write(result[i].astype(str))
        #     fp.write('\n')

    else:
        kernel, degree = args.kernel, args.degree
        C = 0
        resultfile = pd.DataFrame(
            columns=['C', 'Precision', 'Recall', 'Fscore'])

        for i in range(1, 500):
            C = float(i) / 10
            if(args.spearman == False):
                sgd = SGD(args.base_estimator, args.n_estimators, args.lr, args.C, f"../../data/ML/{args.feature}_dimension/train.csv",
                          f"../../data/ML/{args.feature}_dimension/{rate}valid.csv", f"../../data/ML/{args.feature}_dimension/test_b.csv")
            else:
                sgd = SGD(args.base_estimator, args.n_estimators, args.lr, args.C, f"../../data/ML/spearman_selected/train.csv",
                          f"../../data/ML/spearman_selected/{rate}valid.csv", f"../../data/ML/spearman_selected/test.csv")

            sgd.fit()
            P, R, Fscore = sgd.P(), sgd.R(), sgd.Fscore()
            print(f"[C = {C}]\tPrecision: {P}\tRecall: {R}\tFscore: {Fscore}\n")

            resultfile.loc[i] = (str(C), str(P), str(R), str(Fscore))

        resultfile.to_csv(
            f"spearman({args.spearman})_{args.feature}_{args.clf}_{args.kernel}_auto.csv")
