import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FriendlyBlock(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding):
        super(FriendlyBlock, self).__init__()
        self.conv1h = nn.Conv2d(inp, inp, kernel_size=(1, kernel), stride=stride, padding=(0,padding), groups=inp)
        self.conv1v = nn.Conv2d(inp, inp, kernel_size=(kernel, 1), stride=stride, padding=(padding,0), groups=inp)
        self.bn1    = nn.BatchNorm2d(2*inp)

        self.conv2  = nn.Conv2d(2*inp, oup, kernel_size=1)
        self.bn2    = nn.BatchNorm2d(oup)
    
    def forward(self,x):
        out1 = self.conv1h(x)
        out2 = self.conv1v(x)
        out = torch.cat([out1,out2],1)
        out = self.bn1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class AlexNetFriendly(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetFriendly, self).__init__()

        self.c1 = FriendlyBlock(3, 64, 11, 4, 4)
        self.c2 = FriendlyBlock(64, 192,  5, 1, 2)
        self.c3 = FriendlyBlock(192, 384,  3, 1, 1)
        self.c4 = FriendlyBlock(384, 256,  3, 1, 1)
        self.c5 = FriendlyBlock(256, 256,  3, 1, 1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.c1(x)
        x = self.maxpool(x)
        x = self.c2(x)
        x = self.maxpool(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def test():
    net=AlexNet()
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    n2 = AlexNetFriendly()
    z = n2(x) 
    print(y.shape, z.shape, net, n2)
#test()
