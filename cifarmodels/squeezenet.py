'''Squueze Net in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out

class FireFriendly(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(FireFriendly, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, 2*expand_planes-2*squeeze_planes, kernel_size=1, stride=1)
        self.conv3_h = nn.Conv2d(squeeze_planes, squeeze_planes, kernel_size=(1,3), stride=1, padding=(0,1), groups=squeeze_planes)
        self.conv3_d = nn.Conv2d(squeeze_planes, squeeze_planes, kernel_size=(3,1), stride=1, padding=(1,0), groups=squeeze_planes)
        self.bn2 = nn.BatchNorm2d(2*expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        out1 = self.conv2(x)
        out2 = torch.cat([self.conv3_h(x), self.conv3_d(x)], 1)
        out = self.bn2(torch.cat([out1, out2],1))
        out = self.relu2(out)
        return out


class SqueezeNetwork(nn.Module):
    def __init__(self, block, num_classes):
        super(SqueezeNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = block(64, 16, 64)
        self.fire3 = block(128, 16, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire4 = block(128, 32, 128)
        self.fire5 = block(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire6 = block(256, 48, 192)
        self.fire7 = block(384, 48, 192)
        self.fire8 = block(384, 64, 256)
        self.fire9 = block(512, 64, 256)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.fire2(x)
        x = self.fire3(x)      
        x = self.maxpool2(x)

        x = self.fire4(x)
        x = self.fire5(x)    
        x = self.maxpool3(x)

        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

def SqueezeNet(num_classes=100):
    return SqueezeNetwork(Fire, num_classes)

def SqueezeNetFriendly(num_classes=100):
    return SqueezeNetwork(FireFriendly, num_classes)

def test():
    net = SqueezeNet()
    x = torch.randn(4, 3, 32, 32)
    y = net(x)
    net = SqueezeNetFriendly()
    x = torch.randn(4, 3, 32, 32)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()