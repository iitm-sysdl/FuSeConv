'''
FuSeConv: Fully Separable Convolutions for Fast Inference on Systolic Arrays
Authors: Surya Selvam, Vinod Ganesan, Pratyush Kumar
Email ID: selvams@purdue.edu, vinodg@cse.iitm.ac.in, pratyush@cse.iitm.ac.in
'''
import torch
import torch.nn as nn
import torch.nn.init as init

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)

        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)

        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class FireFriendly(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(FireFriendly, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes+expand3x3_planes-2*squeeze_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand1x3 = nn.Conv2d(squeeze_planes, squeeze_planes, kernel_size=(1,3), padding=(0,1), groups=squeeze_planes)
        self.expand3x1 = nn.Conv2d(squeeze_planes, squeeze_planes, kernel_size=(1,3), padding=(0,1), groups=squeeze_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.squeeze_activation((self.squeeze(x)))
        out1 = self.expand1x1_activation((self.expand1x1(out)))
        out2 = self.expand3x3_activation((torch.cat( [self.expand1x3(out), self.expand3x1(out)], 1) ))
        return torch.cat([out1, out2], 1)

class Squeeze(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(Squeeze, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            block(64, 16, 64, 64),
            block(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            block(128, 32, 128, 128),
            block(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            block(256, 48, 192, 192),
            block(384, 48, 192, 192),
            block(384, 64, 256, 256),
            block(512, 64, 256, 256),
        )
        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

def SqueezeNet(num_classes=100):
    return Squeeze(Fire, num_classes)

def SqueezeNetFriendly(num_classes=100):
    return Squeeze(FireFriendly, num_classes)

def test():
    net = SqueezeNet()
    y = net(torch.randn(1,3,224,224))
    net = SqueezeNetFriendly()
    y = net(torch.randn(1,3,224,224))
    print(y.size())

if __name__ == '__main__':
    test()        
