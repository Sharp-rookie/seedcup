import pandas as pd
import numpy as np


class MLTrainSeedDataset:

    def __init__(self, annotations_file):
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.data: pd.DataFrame = self.data[self.data['label'].notna()]
        self.Y = self.data['label']
        self.X = self.data.drop(['id', 'label'], axis=1).fillna(value=-1)
        self.X = np.array(self.X).tolist()
        self.Y = list(np.array(self.Y).astype(np.int64))

    def __len__(self):
        return len(self.data)


class MLTestSeedDataset:
    def __init__(self, label_file):
        self.data: pd.DataFrame = pd.read_csv(label_file)
        self.X = self.data.drop(['id'], axis=1).fillna(value=-1)
        self.X = np.array(self.X).tolist()
