'''
FuSeConv: Fully Separable Convolutions for Fast Inference on Systolic Arrays
Authors: Surya Selvam, Vinod Ganesan, Pratyush Kumar
Email ID: selvams@purdue.edu, vinodg@cse.iitm.ac.in, pratyush@cse.iitm.ac.in
'''
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
        self.conv1h = nn.Conv2d(in_planes//2, in_planes//2, kernel_size=(1, 3), padding=(0,1), groups=in_planes//2)
        self.bn1h   = nn.BatchNorm2d(in_planes//2) 
        self.conv1v = nn.Conv2d(in_planes//2, in_planes//2, kernel_size=(3, 1), padding=(1,0), groups=in_planes//2)
        self.bn1v   = nn.BatchNorm2d(in_planes//2)

        self.conv2  = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.bn2    = nn.BatchNorm2d(out_planes)
    
    def forward(self,x):
        out1, out2 = x.chunk(2,1)
        out1 = self.bn1h(self.conv1h(out1))
        out2 = self.bn1v(self.conv1v(out2))
        out = torch.cat([out1,out2],1)
        out = F.relu(self.bn2(self.conv2(out)))
        return out

# Removed Big FC Layer
class VGGClass(nn.Module):
    def __init__(self, vgg_name, block, numClasses):
        super(VGGClass, self).__init__()
        self.features = self._make_layers(block, cfg[vgg_name])
        self.classifier = nn.Sequential( 
                nn.Dropout(), 
                nn.Linear(512,numClasses),
                )
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, block, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if in_channels == 3:
                    layers.append(Block(in_channels, x))
                else:
                    layers.append(block(in_channels,x))
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1,1))]
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
    
def VGG(num_classes=100):
    return VGGClass('VGG16', Block, num_classes)

def VGGFriendly(num_classes=100):
    return VGGClass('VGG16', FriendlyBlock, num_classes)

def test():
    net = VGG()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = VGGFriendly()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
