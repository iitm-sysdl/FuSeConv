'''
FuSeConv: Fully Separable Convolutions for Fast Inference on Systolic Arrays
Authors: Surya Selvam, Vinod Ganesan, Pratyush Kumar
Email ID: selvams@purdue.edu, vinodg@cse.iitm.ac.in, pratyush@cse.iitm.ac.in
'''
'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class FriendlyBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(FriendlyBlock, self).__init__()
        self.conv1_h = nn.Conv2d(in_planes//2, in_planes//2, kernel_size=(1,3), stride=stride, padding=(0,1), groups=in_planes//2, bias=False)
        self.bn1_h = nn.BatchNorm2d(in_planes//2)
        self.conv1_v = nn.Conv2d(in_planes//2, in_planes//2, kernel_size=(3,1), stride=stride, padding=(1,0), groups=in_planes//2, bias=False)
        self.bn1_v = nn.BatchNorm2d(in_planes//2)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out1, out2 = x.chunk(2,1)
        out1 = self.bn1_h(self.conv1_h(out1))
        out2 = self.bn1_v(self.conv1_v(out2))
        out  = torch.cat([out1, out2], 1)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class FriendlyBlock2(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(FriendlyBlock2, self).__init__()
        self.conv1_h = nn.Conv2d(in_planes, in_planes, kernel_size=(1,3), stride=stride, padding=(0,1), groups=in_planes, bias=False)
        self.bn1_h = nn.BatchNorm2d(in_planes)
        self.conv1_v = nn.Conv2d(in_planes, in_planes, kernel_size=(3,1), stride=stride, padding=(1,0), groups=in_planes, bias=False)
        self.bn1_v = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(2*in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out1 = self.bn1_h(self.conv1_h(x))
        out2 = self.bn1_v(self.conv1_v(x))
        out  = torch.cat([out1, out2], 1)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    depth_mul = 1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, block, num_classes):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(block, in_planes=32)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(int(1024 * self.depth_mul), num_classes)

    def _make_layers(self, block, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = int(x * self.depth_mul) if isinstance(x, int) else int(x[0]*self.depth_mul)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def MobileNetV1(num_classes=1000):
    return MobileNet(Block, num_classes)

def MobileNetV1Friendly(num_classes=1000):
    return MobileNet(FriendlyBlock, num_classes)

def MobileNetV1Friendly2(num_classes=1000):
    return MobileNet(FriendlyBlock2, num_classes)

def test():
    net = MobileNetV1()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV1Friendly()
    x = torch.randn(1,3,128,128)
    y = net(x)
    print(y.size())
    x = torch.randn(1,3,288,288)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()