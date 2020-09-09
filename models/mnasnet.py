import torch.nn as nn
import torch
import math

def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def Conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def SepConv_3x3(inp, oup): #input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp , 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

class SepConvFriendly(nn.Module):
    def __init__(self, inp, oup):
        super(SepConvFriendly, self).__init__()

        self.conv2_h = nn.Conv2d(inp//2, inp//2, (1, 3), 1, (0, 1), groups=inp//2, bias=False)
        self.bn2_h   = nn.BatchNorm2d(inp//2)
        self.conv2_v = nn.Conv2d(inp//2, inp//2, (3, 1), 1, (1, 0), groups=inp//2, bias=False)
        self.bn2_v   = nn.BatchNorm2d(inp//2)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)

    def forward(self, x):        
        out1, out2 = x.chunk(2,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out1)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out2)))
        out = torch.cat([out1, out2], 1)
        out = self.bn3(self.conv3(out))
        return out

class SepConvFriendly2(nn.Module):
    def __init__(self, inp, oup):
        super(SepConvFriendly2, self).__init__()

        self.conv2_h = nn.Conv2d(inp, inp, (1,3), 1, (0,1), groups=inp, bias=False)
        self.bn2_h   = nn.BatchNorm2d(inp)
        self.conv2_v = nn.Conv2d(inp, inp, (3,1), 1, (1,0), groups=inp, bias=False)
        self.bn2_v   = nn.BatchNorm2d(inp)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(2*inp, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)


    def forward(self, x):        
        out1 = self.r2(self.bn2_h(self.conv2_h(x)))
        out2 = self.r2(self.bn2_v(self.conv2_v(x)))
        out = torch.cat([out1, out2], 1)
        out = self.bn3(self.conv3(out))
        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidualFriendly(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidualFriendly, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = round(inp * expand_ratio)

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim//2, hidden_dim//2, (1, kernel), stride, (0, kernel//2), groups=hidden_dim//2, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim//2)
        self.conv2_v = nn.Conv2d(hidden_dim//2, hidden_dim//2, (kernel, 1), stride, (kernel//2, 0), groups=hidden_dim//2, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim//2)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)

    def forward(self, x):
        out = self.r1(self.bn1(self.conv1(x)))
        
        out1, out2 = out.chunk(2,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out1)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out2)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.use_res_connect:
            return x + out
        else:
            return out

class InvertedResidualFriendly2(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidualFriendly2, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = round(inp * expand_ratio)

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim, hidden_dim, (1, kernel), stride, (0, kernel//2), groups=hidden_dim, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim)
        self.conv2_v = nn.Conv2d(hidden_dim, hidden_dim, (kernel, 1), stride, (kernel//2, 0), groups=hidden_dim, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(2*hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)

    def forward(self, x):
        out = self.r1(self.bn1(self.conv1(x)))
        
        out1 = self.r2(self.bn2_h(self.conv2_h(out)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.use_res_connect:
            return x + out
        else:
            return out


class MnasNetClass(nn.Module):
    def __init__(self, block, n_class=1000, input_size=224, width_mult=1.):
        super(MnasNetClass, self).__init__()

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [3, 24,  3, 2, 3],  # -> 56x56
            [3, 40,  3, 2, 5],  # -> 28x28
            [6, 80,  3, 2, 5],  # -> 14x14
            [6, 96,  2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # building first two layer
        if block == InvertedResidual:
            self.features = [Conv_3x3(3, input_channel, 2), SepConv_3x3(input_channel, 16)]
        elif block == InvertedResidualFriendly:
            self.features = [Conv_3x3(3, input_channel, 2), SepConvFriendly(input_channel, 16)]
        elif block == InvertedResidualFriendly2:
            self.features = [Conv_3x3(3, input_channel, 2), SepConvFriendly2(input_channel, 16)]
        input_channel = 16

        # building inverted residual blocks (MBConv)
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, t, k))
                else:
                    self.features.append(block(input_channel, output_channel, 1, t, k))
                input_channel = output_channel

        # building last several layers
        self.features.append(Conv_1x1(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool2d(1))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def MnasNet(num_classes=1000):
    return MnasNetClass(InvertedResidual, num_classes)

def MnasNetFriendly(num_classes=1000):
    return MnasNetClass(InvertedResidualFriendly, num_classes)

def MnasNetFriendly2(num_classes=1000):
    return MnasNetClass(InvertedResidualFriendly2, num_classes)

def test():
    net = MnasNet()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MnasNetFriendly()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    net = MnasNetFriendly2()
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())
    

if __name__ == '__main__':
    test()