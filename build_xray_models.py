# import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(Residual_block, self).__init__()
        self.essential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = shortcut

    def forward(self, x):
        out = self.essential(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)


def make_layer(in_channels, out_channels, num_block, stride=1):
    """

    :param in_channels:
    :param out_channels:
    :param num_block:
    :param stride: the stride of the first residual block in a layer
    :return :
    """
    layers = []
    if stride != 1:
        # dotted line skip, indicating the increase of dimension/channel
        shortcut = nn.Sequential(
            # k=1,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
    else:
        # solid line skip
        shortcut = None
    layers.append(Residual_block(in_channels, out_channels, stride, shortcut))
    # solid line skip
    for i in range(1, num_block):
        layers.append(Residual_block(out_channels, out_channels))

    return nn.Sequential(*layers)


class Resnet(nn.Module):
    """
    the input tensor data should be [batch_size, channel, 224, 224]
    """
    def __init__(self, in_channel, num_labels=3):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            # (224+2p-k)//2 + 1 = c, k=7, c=112, so p=3
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # process the first maxpool separately, then it's convenient to divide the residual block
            # (112+2p-k)//2 + 1 = c, k=3, c=56, so p=1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 56*56*64 -> 56*56*64, s=1
        self.layer1 = make_layer(64, 64, num_block=3)
        # 56*56*64 -> 28*28*128, s=2
        self.layer2 = make_layer(64, 128, num_block=4, stride=2)
        # 28*28*128 -> 14*14*256, s=2
        self.layer3 = make_layer(128, 256, num_block=6, stride=2)
        # 14*14*256 -> 7*7*512, s=2
        self.layer4 = make_layer(256, 512, num_block=3, stride=2)

        # dense
        self.fc = nn.Linear(512, num_labels)

    def forward(self, x):
        xx = self.pre(x)

        xx = self.layer1(xx)
        xx = self.layer2(xx)
        xx = self.layer3(xx)
        xx = self.layer4(xx)

        # pool = nn.AvgPool2d(kernel_size=7)
        # xx = pool(xx)
        xx = F.avg_pool2d(xx, kernel_size=7)

        # print(xx.shape)
        xx = xx.view(xx.shape[0], -1)
        # print(xx.shape)

        return self.fc(xx)