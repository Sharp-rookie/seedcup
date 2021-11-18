import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SVM:
    def __init__(self, clf, kernel, C, degree, trainfile, validfile, testfile):
        super(SVM, self).__init__()

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

        if(clf == "SVC"):
            if(kernel == "rbf"):
                self.classifier = Pipeline([
                    # ('pca',PCA()),
                    ("scaler", StandardScaler()),
                    ("svm_clf", SVC(kernel="rbf", gamma='auto', C=C))])
            elif(kernel == "poly"):
                self.classifier = Pipeline([
                    # ('pca',PCA()),
                    ("scaler", StandardScaler()),
                    ("svm_clf", SVC(kernel="poly", degree=degree, coef0=1, C=C))])
            elif(kernel == "sigmoid"):
                self.classifier = Pipeline([
                    # ('pca',PCA()),
                    ("scaler", StandardScaler()),
                    ("svm_clf", SVC(kernel="sigmoid", degree=degree, C=C))])

        elif(clf == "NuSVC"):
            # NuSVC 不需要参数 C
            if(kernel == "rbf"):
                self.classifier = Pipeline([
                    # ('pca',PCA()),
                    ("scaler", StandardScaler()),
                    ("svm_clf", NuSVC(kernel="rbf", gamma='auto'))])
            elif(kernel == "poly"):
                self.classifier = Pipeline([
                    # ('pca',PCA()),
                    ("scaler", StandardScaler()),
                    ("svm_clf", NuSVC(kernel="poly", degree=degree, coef0=1))])
            elif(kernel == "sigmoid"):
                self.classifier = Pipeline([
                    # ('pca',PCA()),
                    ("scaler", StandardScaler()),
                    ("svm_clf", NuSVC(kernel="sigmoid", degree=degree))])

    def fit(self):
        self.classifier.fit(self.train_x, self.train_y)

    def P(self):
        index_ = self.classifier.predict(self.valid_x) == 1
        TP = (self.valid_y[index_] == 1).sum()
        if(index_.sum() == 0):
            return 0

        return round(TP / index_.sum(), 4)

    def P1(self):
        index_ = self.classifier.predict(self.valid_x) == 1
        TP = (self.valid_y[index_] == 1).sum()
        if(index_.sum() == 0):
            return 0

        return round(TP / index_.sum(), 4)

    def P0(self):
        index_ = self.classifier.predict(self.valid_x) == 0
        TP = (self.valid_y[index_] == 0).sum()
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
