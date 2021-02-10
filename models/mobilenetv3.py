'''
FuSeConv: Fully Separable Convolutions for Fast Inference on Systolic Arrays
Authors: Surya Selvam, Vinod Ganesan, Pratyush Kumar
Email ID: selvams@purdue.edu, vinodg@cse.iitm.ac.in, pratyush@cse.iitm.ac.in
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

class Flatten(nn.Module):
  def forward(self, x):
    N, C, H, W = x.size() # read in N, C, H, W
    return x.view(N, -1)

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
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        
        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileBottleneckHalf(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneckHalf, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv_layer(inp, exp, 1, 1, 0, bias=False)
        self.bn1 = norm_layer(exp)
        self.nl1 = nlin_layer(inplace=True)
		
        self.conv2_h = conv_layer(exp//2, exp//2, kernel_size=(1, kernel),stride=stride, padding=(0, padding), groups=exp//2, bias=False)
        self.bn2_h = norm_layer(exp//2)
        self.conv2_v = conv_layer(exp//2, exp//2, kernel_size=(kernel, 1),stride=stride, padding=(padding, 0), groups=exp//2, bias=False)
        self.bn2_v = norm_layer(exp//2)
        self.se1 = SELayer(exp)
        self.nl2 = nlin_layer(inplace=True)
        
        self.conv3 = conv_layer(exp, oup, 1, 1, 0, bias=False)
        self.bn3 = norm_layer(oup)
	 
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nl1(out)
        
        out1, out2 = out.chunk(2,1)
        out1 = self.bn2_h(self.conv2_h(out1))
        out2 = self.bn2_v(self.conv2_v(out2))
        out = torch.cat([out1, out2], 1)
        out = self.se1(out)
        out = self.nl2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_res_connect:
            return x + out
        else:
            return out
    
class MobileBottleneckFull(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneckFull, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv_layer(inp, exp, 1, 1, 0, bias=False)
        self.bn1 = norm_layer(exp)
        self.nl1 = nlin_layer(inplace=True)
		
        self.conv2_h = conv_layer(exp, exp, kernel_size=(1, kernel),stride=stride, padding=(0, padding), groups=exp, bias=False)
        self.bn2_h = norm_layer(exp)
        self.conv2_v = conv_layer(exp, exp, kernel_size=(kernel, 1),stride=stride, padding=(padding, 0), groups=exp, bias=False)
        self.bn2_v = norm_layer(exp)
        self.se1 = SELayer(2*exp)
        self.nl2 = nlin_layer(inplace=True)
        
        self.conv3 = conv_layer(2*exp, oup, 1, 1, 0, bias=False)
        self.bn3 = norm_layer(oup)
	 
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nl1(out)
        
        # out1, out2 = out.chunk(2,1)
        out1 = self.bn2_h(self.conv2_h(out))
        out2 = self.bn2_v(self.conv2_v(out))
        out = torch.cat([out1, out2], 1)
        out = self.se1(out)
        out = self.nl2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_res_connect:
            return x + out
        else:
            return out


class MobileNetV3Class(nn.Module):
    def __init__(self, block, n_class, mode, dropout=0.20, width_mult=1.0):
        super(MobileNetV3Class, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
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
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(block(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
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
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class MobileNetV3ClassHybrid(nn.Module):
    def __init__(self, block, n_class, mode, dropout=0.20, width_mult=1.0):
        super(MobileNetV3ClassHybrid, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s, fuse
                [3, 16,  16,  False, 'RE', 1, 0],
                [3, 64,  24,  False, 'RE', 2, 1],
                [3, 72,  24,  False, 'RE', 1, 1],
                [5, 72,  40,  True,  'RE', 2, 1],
                [5, 120, 40,  True,  'RE', 1, 1],
                [5, 120, 40,  True,  'RE', 1, 1],
                [3, 240, 80,  False, 'HS', 2, 1],
                [3, 200, 80,  False, 'HS', 1, 1],
                [3, 184, 80,  False, 'HS', 1, 0],
                [3, 184, 80,  False, 'HS', 1, 0],
                [3, 480, 112, True,  'HS', 1, 0],
                [3, 672, 112, True,  'HS', 1, 0],
                [5, 672, 160, True,  'HS', 2, 0],
                [5, 960, 160, True,  'HS', 1, 0],
                [5, 960, 160, True,  'HS', 1, 0],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s, fuse
                [3, 16,  16,  True,  'RE', 2, 0],
                [3, 72,  24,  False, 'RE', 2, 0],
                [3, 88,  24,  False, 'RE', 1, 1],
                [5, 96,  40,  True,  'HS', 2, 0],
                [5, 240, 40,  True,  'HS', 1, 1],
                [5, 240, 40,  True,  'HS', 1, 1],
                [5, 120, 48,  True,  'HS', 1, 1],
                [5, 144, 48,  True,  'HS', 1, 1],
                [5, 288, 96,  True,  'HS', 2, 0],
                [5, 576, 96,  True,  'HS', 1, 0],
                [5, 576, 96,  True,  'HS', 1, 0],
            ]
        else:
            raise NotImplementedError

        # building first layer
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s ,fuse in mobile_setting:
            if fuse==0:
                blockHybrid=MobileBottleneck
            else:
                blockHybrid=block
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(blockHybrid(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
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
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def MobileNetV3(mode='small', num_classes=1000):
    return MobileNetV3Class(MobileBottleneck, num_classes, mode)

def MobileNetV3FuSeHalf(mode='small', num_classes=1000):
    return MobileNetV3Class(MobileBottleneckHalf, num_classes, mode)

def MobileNetV3FuSeFull(mode='small', num_classes=1000):
    return MobileNetV3Class(MobileBottleneckFull, num_classes, mode)

def MobileNetV3FuSeHalfHybrid(mode='small', num_classes=1000):
    return MobileNetV3ClassHybrid(MobileBottleneckHalf, num_classes, mode)

def MobileNetV3FuSeFullHybrid(mode='small', num_classes=1000):
    return MobileNetV3ClassHybrid(MobileBottleneckFull, num_classes, mode)

def test():
    net = MobileNetV3()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV3FuSeHalf()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV3FuSeFull()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV3FuSeHalfHybrid()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV3FuSeFullHybrid()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
