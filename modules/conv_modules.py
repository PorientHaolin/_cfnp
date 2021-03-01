import torch.nn.functional as F
import torch
import torch.nn as nn


class Basic_Block(nn.Module):
    '''Res18基础块'''
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(Basic_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN_Res18(nn.Module):
    def __init__(self, block, num_blocks, num_generate, num_channels=1):
        super(CNN_Res18, self).__init__()

        self.layer4_channels = num_generate
        self.layer3_channels = int(self.layer4_channels / 2)
        self.layer2_channels = int(self.layer3_channels / 2)
        self.layer1_channels = int(self.layer2_channels / 2)

        self.in_channels = self.layer1_channels
        self.conv1 = nn.Conv2d(num_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=(2, 1))
        self.layer1 = self._make_layer(block, self.layer1_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.layer2_channels, num_blocks[1], stride=(2, 1))
        self.layer3 = self._make_layer(block, self.layer3_channels, num_blocks[2], stride=(2, 1))
        self.layer4 = self._make_layer(block, self.layer4_channels, num_blocks[3], stride=(2, 1))
        # self.in1

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=(4, 3), padding=(2, 1), stride=(2, 1))
        out = torch.mean(out, dim=2)
        out = torch.mean(out, dim=0)
        return out
