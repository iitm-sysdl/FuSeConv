'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.bn(self.conv1(x))
        return x

class FriendlyBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes):
        super(FriendlyBlock, self).__init__()
        self.conv1h = nn.Conv2d(in_planes, in_planes, kernel_size=(1, 3), padding=(0,1), groups=in_planes)
        self.conv1v = nn.Conv2d(in_planes, in_planes, kernel_size=(3, 1), padding=(1,0), groups=in_planes)
        self.bn1    = nn.BatchNorm2d(2*in_planes)

        self.conv2  = nn.Conv2d(2*in_planes, out_planes, kernel_size=1)
        self.bn2    = nn.BatchNorm2d(out_planes)
    
    def forward(self,x):
        out1 = self.conv1h(x)
        out2 = self.conv1v(x)
        out = torch.cat([out1,out2],1)
        out = self.bn1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential( 
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True), nn.Dropout(), nn.Linear(4096,4096),
                nn.ReLU(True), nn.Dropout(), nn.Linear(4096,1000),
                )
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(Block(in_channels,x))
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((7,7))]
        return nn.Sequential(*layers)
    
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
    
class VGGFriendly(nn.Module):
    def __init__(self, vgg_name):
        super(VGGFriendly, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential( 
                nn.Linear(512*7*7, 4096),
                nn.ReLU(True), nn.Dropout(), nn.Linear(4096,4096),
                nn.ReLU(True), nn.Dropout(), nn.Linear(4096,1000),
                )
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(FriendlyBlock(in_channels,x))
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((7,7))]
        return nn.Sequential(*layers)

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
    n1 = VGG('VGG16')
    n2 = VGGFriendly('VGG16')
    x = torch.randn(1, 3, 224, 224)
    y = n1(x)
    z = n2(x)
    print(y.shape, z.shape, n1, n2)

#test()

