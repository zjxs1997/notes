# ResNet
resenet应该是15年提出来的工作。说到这个名字，就可以联想到残差连接。比较常见的有两种，分别是resent50和101。
可以认为resnet主要有5个stage，第一个stage都是一个7 * 7卷积加一个3 * 3的maxpooling。后面4个stage每个stage都是几个小卷积的集合sublayer（或者说block），每个sublayer重复若干次。然后每次有经过sublayer的路径和不经过直接送过去的残差连接。
而50和101的区别就在于stage4中sublayer的重复次数。

输入的形状貌似和VGG一样，也是(3, 224, 224)的图片。

试着写了一下resnet-50的代码
```python
from typing import Tuple
import torch
import torch.nn as nn

# 是每个stage的第一个block，除了stage2之外，都会让feature数量减少
class BasicBlock(nn.Module):
    def __init__(self, 
        in_channels: int, 
        channels: Tuple[int, int, int], 
        stride: int
    ):
        super().__init__()
        c1, c2, c3 = channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, c1, 1, stride=stride, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 3, padding=True, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c3, 1, bias=False),
            nn.BatchNorm2d(c3)
        )
        self.shortcut = nn.Conv2d(in_channels, c3, 1, stride=stride, bias=False)
        self.batchnorm = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x_shortcut = self.batchnorm(x_shortcut)
        output = self.block(x)
        output = output + x_shortcut
        output = self.relu(output)
        return output


# 这个模块的stride是1，不会让feature数量缩小
class IdenticalBlock(nn.Module):
    def __init__(self, in_channels: int, channels: Tuple[int, int, int]):
        super().__init__()
        c1, c2, c3 = channels
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=True, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c3, kernel_size=1, bias=False),
            nn.BatchNorm2d(c3),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x_shortcut = x
        output = self.block(x)
        output = output + x_shortcut
        output = self.relu(output)
        return output


class Resnet(nn.Module):
    def __init__(self, output_dim=384):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        stage2_channels = [64,64,256]
        stage3_channels = [128,128,512]
        stage4_channels = [256,256,1024]
        stage5_channels = [512,512,2048]

        stage2_list = [BasicBlock(64, stage2_channels, 1),] + [IdenticalBlock(256, stage2_channels)] * 2
        self.stage2 = nn.Sequential(*stage2_list)
        stage3_list = [BasicBlock(256, stage3_channels, 2)] + [IdenticalBlock(512, stage3_channels)] * 3
        self.stage3 = nn.Sequential(*stage3_list)
        stage4_list = [BasicBlock(512, stage4_channels, 2)] + [IdenticalBlock(1024, stage4_channels)] * 5
        self.stage4 = nn.Sequential(*stage4_list)
        stage5_list = [BasicBlock(1024, stage5_channels, 2)] + [IdenticalBlock(2048, stage5_channels)] * 2
        self.stage5 = nn.Sequential(*stage5_list)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, output_dim)

    def forward(self, x):
        output = self.stage1(x)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.stage5(output)

        output = self.pooling(output)
        output = torch.flatten(output, 1)
        output = self.linear(output)
        return output
```

这个resnet的参数量只有14+m左右，比vgg低了一个数量级。


