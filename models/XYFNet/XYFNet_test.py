# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         conv_test
# Description:
# Author:       梁超
# Date:         2021/10/23
# -------------------------------------------------------------------------------
import torch

from XYFNet_model import CTNetPlus
# from res_model import *
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="path to model",
                        type=str, default="../../checkpoints/XYFNet/146_epoc_loss_0.5038768845214955_fscore_0.8025766085680138.pt")
    parser.add_argument('-i', '--input', help="path to input files",
                        type=str, default="../../data/33_dimension/test_b.csv")
    parser.add_argument(
        '-o', '--output', help="path to output files", type=str, default="XYF_output_0.8257.txt")
    parser.add_argument('--input-features',
                        help="input dimension for model", type=int, default=33)
    parser.add_argument('--output-features',
                        help="output dimension for model", type=int, default=2)

    return parser.parse_args()


class SeedDataset(Dataset):

    def __init__(self, annotations_file):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.X = self.data.drop(columns=['id']).fillna(value=-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.X.iloc[idx].values).type(torch.FloatTensor)


def main():
    args = parse_args()
    # model = Fake1DAttention(args.input_features, args.output_features)
    model = CTNetPlus(batch=1, in_channels=33, out_channels=2)
    model.load_state_dict(torch.load(args.model))

    model.eval()

    test_dataset = SeedDataset(args.input)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    outputs = []
    for x in test_dataloader:
        logit = model(x)
        outputs.append(str(logit.argmax(1).item()))

    with open(args.output, 'w') as f:
        f.write('\n'.join(outputs))


if __name__ == "__main__":
    main()
