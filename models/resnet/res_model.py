import torch
import torch.nn as nn


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=None):
        super(ResidualBlock, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.down_sample = down_sample
        self.fc1 = nn.Linear(self.in_features, 2 * self.in_features)
        self.bn1 = nn.BatchNorm1d(2 * self.in_features)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(2 * self.in_features, self.out_features)
        self.bn2 = nn.BatchNorm1d(self.out_features)
        self.activation2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        if self.down_sample:
            residual = self.down_sample(residual)
        out += residual
        out = self.activation2(out)
        return out

# ResNet


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Linear(self.in_channels, self.in_channels)
        self.bn = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(
            block, self.in_channels, 2*self.in_channels, layers[0])
        self.layer2 = self.make_layer(
            block, 2*self.in_channels, 4*self.in_channels, layers[1])
        self.layer3 = self.make_layer(
            block, 4*self.in_channels, self.in_channels, layers[2])
        self.fc = nn.Linear(self.in_channels, num_classes)
        self.last = nn.Softmax(dim=1)

    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        down_sample = None
        if in_channels != out_channels:
            down_sample = nn.Sequential(nn.Linear(in_channels, out_channels),
                                        nn.BatchNorm1d(out_channels))
        layers.append(block(in_channels, out_channels, down_sample))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.last(out)
        return out


if __name__ == '__main__':
    batch, in_features, out_features = 30, 28, 2
    model = ResNet(ResidualBlock, [2, 2, 2])
    x = torch.randn(batch, in_features)
    print(model(x).shape)
