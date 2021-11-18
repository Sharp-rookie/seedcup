# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         conv_model
# Description:
# Author:       梁超
# Date:         2021/10/23
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn


class CTNetPlus(nn.Module):
    def __init__(self, batch, in_channels, out_channels):
        super(CTNetPlus, self).__init__()
        self.batch = batch
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc1 = nn.Linear(self.in_channels, 2 * self.in_channels)
        self.conv1 = nn.Conv1d(1, 16, (3,))
        self.mp1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(16, self.in_channels-1)

        self.fc2 = nn.Linear(self.in_channels-1, self.in_channels-1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=(3,),
                               padding=(1,), stride=(1,))
        self.mp2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(32, 16)

        self.pt = nn.Flatten()
        self.fc3 = nn.Linear(32*16, 2)
        self.act = nn.LeakyReLU()
        self.pre = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, self.in_channels))
        x = self.fc1(x)
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.bn2(x)
        x = self.pt(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.pre(x)
        return x


if __name__ == '__main__':
    batch, in_features, out_features = 30, 33, 2
    model = CTNetPlus(batch, in_features, out_features)
    x = torch.randn(batch, in_features)
    print(model(x).shape)
