import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = [ # k, exp, c,  se,     nl,  s,
        [3, 64,  24,  False, 'RE', 2],
        [3, 72,  24,  False, 'RE', 1],
        [5, 72,  40,  True,  'RE', 2],
        [5, 120, 40,  True,  'RE', 1],
        [5, 120, 40,  True,  'RE', 1],
        [3, 240, 80,  False, 'HS', 2],
        [3, 200, 80,  False, 'HS', 1],
        [3, 184, 80,  False, 'HS', 1],
        [3, 184, 80,  False, 'HS', 1],
        [3, 480, 112, True,  'HS', 1],
        [3, 672, 112, True,  'HS', 1],
        [5, 672, 160, True,  'HS', 2],
        [5, 960, 160, True,  'HS', 1],
        [5, 960, 160, True,  'HS', 1],
    ]


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        reduction_channels = make_divisible(channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = Hsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(DepthwiseSeparableBlock, self).__init__()
        assert inp == oup
        padding = (kernel - 1) // 2

        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish

        self.dconv = nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=padding, groups=inp, bias=False)
        self.bn1   = nn.BatchNorm2d(oup) 
        self.nl    = nlin_layer(inplace=True)
		
        self.pconv = nn.Conv2d(oup, oup, kernel_size=1, stride=1, bias=False)
        self.bn2   = nn.BatchNorm2d(oup)
	 
    def forward(self, x):
        out = self.dconv(x)
        out = self.bn1(out)
        out = self.nl(out)

        out = self.pconv(out)
        out = self.bn2(out)
        return x + out

class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        
        if nl == 'RE':
            nlin_layer = nn.ReLU 
        elif nl == 'HS':
            nlin_layer = Hswish

        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.pconv1 = nn.Conv2d(inp, exp, kernel_size=1, stride=1, bias=False)
        self.bn1    = nn.BatchNorm2d(exp)
        self.nl1    = nlin_layer(inplace=True)

        self.dconv  = nn.Conv2d(exp, exp, kernel_size=kernel, stride=stride, padding=padding, groups=exp, bias=False)
        self.bn2    = nn.BatchNorm2d(exp) 
        self.nl2    = nlin_layer(inplace=True)
        self.se     = SELayer(exp)
        self.pconv2 = nn.Conv2d(exp, oup, kernel_size=1, stride=1, bias=False)
        self.bn3    = nn.BatchNorm2d(oup)
        
    def forward(self, x):
        out = self.pconv1(x)
        out = self.bn1(out)
        out = self.nl1(out)

        out = self.dconv(out)
        out = self.bn2(out)
        out = self.nl2(out)
        out = self.se(out)

        out = self.pconv2(out)
        out = self.bn3(out)
        
        if self.use_res_connect:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, cfg=cfg, dropout=0.20, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280

        self.convstem = [nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(input_channel),
                         Hswish()]
        self.convstem = nn.Sequential(*self.convstem)

        self.features = [DepthwiseSeparableBlock(input_channel, 16, 3, 1, 16, se=False, nl='RE')]    
        input_channel = 16
        
        for k, exp, c, se, nl, s in cfg:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
        self.features = nn.Sequential(*self.features)

        last_conv_channel = make_divisible(960 * width_mult)
        self.tail = [nn.Conv2d(input_channel, last_conv_channel, kernel_size=1, bias=False),
                     nn.BatchNorm2d(last_conv_channel),
                     Hswish(),
                     nn.AdaptiveAvgPool2d(1),
                     nn.Conv2d(last_conv_channel, last_channel, kernel_size=1),
                     Hswish()
        ]
        self.tail = nn.Sequential(*self.tail)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

    def forward(self, x):
        out = self.convstem(x)
        out = self.features(out)
        out = self.tail(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out

def test():
    net = MobileNetV3()
    import timm
    m = timm.create_model('mobilenetv3_large_100', pretrained=True)
    target_state_dict = net.state_dict()
    src_state_dict = m.state_dict()
    target_keys = list(target_state_dict.keys())
    src_keys = list(src_state_dict.keys())
    for i, key in enumerate(target_state_dict):
        target_state_dict[key] = src_state_dict[src_keys[i]]
    net.load_state_dict(target_state_dict)
    y = net(x)
    print(torch.all(torch.eq(y, y2)))
    
if __name__ == '__main__':
    test()
