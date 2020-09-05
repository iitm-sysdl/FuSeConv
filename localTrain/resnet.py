import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckFriendly(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckFriendly, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.conv2_h = nn.Conv2d(planes // 2 , planes //2 , kernel_size = (1,3), stride = stride, padding = (0,1), groups = planes //2, bias=False)
        self.bn2_h = nn.BatchNorm2d(planes // 2 )
        self.conv2_v = nn.Conv2d(planes // 2 , planes //2 , kernel_size = (3,1), stride = stride, padding = (1,0), groups = planes // 2, bias=False)
        self.bn2_v = nn.BatchNorm2d(planes // 2 )

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)
        out1, out2 = out.chunk(2,1)
        out1 = self.conv2_h(out1)
        out1 = self.bn2_h(out1)
        out1 = self.relu(out1)
        out2 = self.conv2_v(out2)
        out2 = self.bn2_v(out2)
        out2 = self.relu(out2)

        out = torch.cat([out1,out2],1)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckFriendly2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckFriendly2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.conv2_h = nn.Conv2d(planes, planes, kernel_size = (1,3), stride = stride, padding = (0,1), groups = planes, bias=False)
        self.bn2_h = nn.BatchNorm2d(planes)
        self.conv2_v = nn.Conv2d(planes, planes, kernel_size = (3,1), stride = stride, padding = (1,0), groups = planes, bias=False)
        self.bn2_v = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(2*planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)
        #out1, out2 = out.chunk(2,1)
        out1 = self.conv2_h(out)
        out1 = self.bn2_h(out1)
        out1 = self.relu(out1)
        out2 = self.conv2_v(out)
        out2 = self.bn2_v(out2)
        out2 = self.relu(out2)

        out = torch.cat([out1,out2],1)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckFriendly3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckFriendly3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.conv2_h = nn.Conv2d(planes // 2 , planes //2 , kernel_size = (1,3), stride = stride, padding = (0,1), groups=planes//2, bias=False)
        self.bn2_h = nn.BatchNorm2d(planes // 2 )
        self.conv2_hh = nn.Conv2d(planes // 2 , planes //2 , kernel_size = (1,3), stride = stride, padding = (0,1), groups = planes // 2, bias=False)
        self.bn2_hh = nn.BatchNorm2d(planes // 2 )

        self.conv2_v = nn.Conv2d(planes // 2 , planes //2 , kernel_size = (3,1), stride = stride, padding = (1,0), groups = planes //2, bias=False)
        self.bn2_v = nn.BatchNorm2d(planes // 2 )
        self.conv2_vv = nn.Conv2d(planes // 2 , planes //2 , kernel_size = (3,1), stride = stride, padding = (1,0), groups = planes // 2, bias=False)
        self.bn2_vv = nn.BatchNorm2d(planes // 2 )

        self.conv3 = nn.Conv2d(2*planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)
        out1, out2 = out.chunk(2,1)
        out11 = self.conv2_h(out1)
        out11 = self.bn2_h(out11)
        out11 = self.relu(out11)

        out12 = self.conv2_hh(out1)
        out12 = self.bn2_hh(out12)
        out12 = self.relu(out12)

        out21 = self.conv2_v(out2)
        out21 = self.bn2_v(out21)
        out21 = self.relu(out21)

        out22 = self.conv2_vv(out2)
        out22 = self.bn2_vv(out22)
        out22 = self.relu(out22)

        out = torch.cat([out11,out12, out21, out22],1)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckFriendly4(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckFriendly4, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.conv2_h = nn.Conv2d(planes, planes, kernel_size = (1,3), stride = stride, padding = (0,1), groups=planes, bias=False)
        self.bn2_h = nn.BatchNorm2d(planes)
        self.conv2_hh = nn.Conv2d(planes, planes, kernel_size = (1,3), stride = stride, padding = (0,1), groups=planes, bias=False)
        self.bn2_hh = nn.BatchNorm2d(planes)

        self.conv2_v = nn.Conv2d(planes, planes, kernel_size = (3,1), stride = stride, padding = (1,0), groups=planes, bias=False)
        self.bn2_v = nn.BatchNorm2d(planes)
        self.conv2_vv = nn.Conv2d(planes, planes, kernel_size = (3,1), stride = stride, padding = (1,0), groups=planes, bias=False)
        self.bn2_vv = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(4*planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)
        #out1, out2 = out.chunk(2,1)
        #Gotcha! - its not planes //2 right. yeah
        out11 = self.conv2_h(out)
        out11 = self.bn2_h(out11)
        out11 = self.relu(out11)

        out12 = self.conv2_hh(out)
        out12 = self.bn2_hh(out12)
        out12 = self.relu(out12)

        out21 = self.conv2_v(out)
        out21 = self.bn2_v(out21)
        out21 = self.relu(out21)

        out22 = self.conv2_vv(out)
        out22 = self.bn2_vv(out22)
        out22 = self.relu(out22)

        out = torch.cat([out11,out12, out21, out22],1)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def init_dist_weights(model):
    # https://arxiv.org/pdf/1706.02677.pdf
    # https://github.com/pytorch/examples/pull/262
    for m in model.modules():
        if isinstance(m, BasicBlock): m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
        if isinstance(m, Bottleneck): m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
        if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, bn0=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if bn0: init_dist_weights(model)
    return model

def resnet50friendly(pretrained=False, bn0=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleneckFriendly, [3, 4, 6, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if bn0: init_dist_weights(model)
    return model


def resnet50friendly2(pretrained=False, bn0=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleneckFriendly2, [3, 4, 6, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if bn0: init_dist_weights(model)
    return model

def resnet50friendly3(pretrained=False, bn0=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleneckFriendly3, [3, 4, 6, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if bn0: init_dist_weights(model)
    return model


def resnet50friendly4(pretrained=False, bn0=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleneckFriendly4, [3, 4, 6, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if bn0: init_dist_weights(model)
    return model




def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
