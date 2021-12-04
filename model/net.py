import torch
import torch.nn as nn

# 网络定义逐层写法
class DP1(nn.Module):
    def __init__(self,):
        super(DP1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
        )
        # 80x80x4 --> 20x20x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 20x20x32 --> 10x10x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 10x10x64 --> 5x5x128
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 5x5x128 --> 1x1x128
        self.out = nn.Linear(128,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


class DP2(nn.Module):
    def __init__(self,):
        super(DP2,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # 80x80x4 --> 40x40x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 40x40x32 --> 20x20x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 20x20x64 --> 10x10x128
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # 10x10x128 --> 5x5x256

        self.out = nn.Linear(5*5*256,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

# net = DP2()
# print(net)