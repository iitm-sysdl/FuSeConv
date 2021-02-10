'''
FuSeConv: Fully Separable Convolutions for Fast Inference on Systolic Arrays
Authors: Surya Selvam, Vinod Ganesan, Pratyush Kumar
Email ID: selvams@purdue.edu, vinodg@cse.iitm.ac.in, pratyush@cse.iitm.ac.in
'''
import torch
from torch import nn
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidualHalf(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualHalf, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim//2, hidden_dim//2, (1,3), stride, (0,1), groups=hidden_dim//2, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim//2)
        self.conv2_v = nn.Conv2d(hidden_dim//2, hidden_dim//2, (3,1), stride, (1,0), groups=hidden_dim//2, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim//2)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)

    def forward(self, x):
        if self.expand_ratio == 1:
            out = x
        else: 
            out = self.r1(self.bn1(self.conv1(x)))

        out1, out2 = out.chunk(2,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out1)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out2)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.identity:
            return x + out
        else:
            return out

class InvertedResidualFull(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualFull, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim, hidden_dim, (1,3), stride, (0,1), groups=hidden_dim, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim)
        self.conv2_v = nn.Conv2d(hidden_dim, hidden_dim, (3,1), stride, (1,0), groups=hidden_dim, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(2*hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)

    def forward(self, x):
        if self.expand_ratio == 1:
            out = x
        else: 
            out = self.r1(self.bn1(self.conv1(x)))

        # out1, out2 = out.chunk(2,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.identity:
            return x + out
        else:
            return out


class MobileNetV2Class(nn.Module):
    def __init__(self,block,num_classes=1000,width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2Class, self).__init__()

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

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
                nn.init.zeros_(m.bias)

    def _forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward

class MobileNetV2ClassHybrid(nn.Module):
    def __init__(self,block,num_classes=1000,width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2ClassHybrid, self).__init__()

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, fuse
                [1, 16, 1, 1, 0],
                [6, 24, 2, 2, 1],
                [6, 32, 3, 2, 1],
                [6, 64, 4, 2, 1],
                [6, 96, 3, 1, 0],
                [6, 160, 3, 2, 0],
                [6, 320, 1, 1,0],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s, f in inverted_residual_setting:
            if f == 0:
                blockHybrid = InvertedResidual
            else:
                blockHybrid = block
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(blockHybrid(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

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
                nn.init.zeros_(m.bias)

    def _forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def MobileNetV2(num_classes=1000):
    return MobileNetV2Class(InvertedResidual, num_classes)

def MobileNetV2FuSeHalf(num_classes=1000):
    return MobileNetV2Class(InvertedResidualHalf, num_classes)

def MobileNetV2FuSeFull(num_classes=1000):
    return MobileNetV2Class(InvertedResidualFull, num_classes)

def MobileNetV2FuSeHalfHybrid(num_classes=1000):
    return MobileNetV2ClassHybrid(InvertedResidualHalf, num_classes)

def MobileNetV2FuSeFullHybrid(num_classes=1000):
    return MobileNetV2ClassHybrid(InvertedResidualFull, num_classes)

def test():
    net = MobileNetV2()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV2FuSeHalf()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV2FuSeFull()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV2FuSeHalfHybrid()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MobileNetV2FuSeFullHybrid()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()