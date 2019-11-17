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
        print(x.shape)
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
        print(x.shape)
        out = self.squeeze_activation((self.squeeze(x)))
        out1 = self.expand1x1_activation((self.expand1x1(out)))
        out2 = self.expand3x3_activation((torch.cat( [self.expand1x3(out), self.expand3x1(out)], 1) ))
        return torch.cat([out1, out2], 1)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
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

class SqueezeNetFriendly(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNetFriendly, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireFriendly(64, 16, 64, 64),
            FireFriendly(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireFriendly(128, 32, 128, 128),
            FireFriendly(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireFriendly(256, 48, 192, 192),
            FireFriendly(384, 48, 192, 192),
            FireFriendly(384, 64, 256, 256),
            FireFriendly(512, 64, 256, 256),
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

def test():
    net = SqueezeNet()
    state_dict = torch.load('../pretrainedmodels/squeezenet.pth')
    net.load_state_dict(state_dict, strict=True)
    x = torch.rand(1,3,224,224)
    y = net(x)
    cfg = [     [64, 16, 64, 64],
            [128, 16, 64, 64],
            [128, 32, 128, 128],
            [256, 32, 128, 128],
            [256, 48, 192, 192],
            [384, 48, 192, 192],
            [384, 64, 256, 256],
            [512, 64, 256, 256],
    ]
    # FireLayers=[3,4,6,7,9,10,11,12]
    # change = ['1', '2']
    # for x in change:
    #     x = int(x)
    #     i = FireLayers[x-1]
    #     net.features[i] = FireFriendly(*cfg[x-1])
    
    # for param in net.parameters():
    #   	param.requires_grad = False
    
    # for x in change:
    #     x = int(x)
    #     i = FireLayers[x-1]
    #     for param in net.features[i].parameters():
    #   	    param.requires_grad = True

    # print(net)
    #for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
#test()
        
        
