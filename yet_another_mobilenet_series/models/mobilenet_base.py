"""Common utilities for mobilenet."""
import abc
import collections
import logging
import functools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.common import add_prefix


def _make_divisible(v, divisor, min_value=None):
    """Make channels divisible to divisor.

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CheckpointModule(nn.Module, metaclass=abc.ABCMeta):
    """Discard mid-result using checkpoint."""

    def __init__(self, use_checkpoint=True):
        super(CheckpointModule, self).__init__()
        self._use_checkpoint = use_checkpoint

    def forward(self, *args, **kwargs):
        from torch.utils.checkpoint import checkpoint
        if self._use_checkpoint:
            return checkpoint(self._forward, *args, **kwargs)
        return self._forward(*args, **kwargs)

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        pass


class Identity(nn.Module):
    """Module proxy for null op."""

    def forward(self, x):
        return x


class Narrow(nn.Module):
    """Module proxy for `torch.narrow`."""

    def __init__(self, dimension, start, length):
        super(Narrow, self).__init__()
        self.dimension = dimension
        self.start = start
        self.length = length

    def forward(self, x):
        return x.narrow(self.dimension, self.start, self.length)


class Swish(nn.Module):
    """Swish activation function.

    See: https://arxiv.org/abs/1710.05941
    NOTE: Will consume much more GPU memory compared with inplaced ReLU.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class HSwish(object):
    """Hard Swish activation function.

    See: https://arxiv.org/abs/1905.02244
    """

    def forward(self, x):
        return x * F.relu6(x + 3, True).div_(6)


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation module.

    See: https://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 n_feature,
                 n_hidden,
                 spatial_dims=[2, 3],
                 active_fn=None):
        super(SqueezeAndExcitation, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.spatial_dims = spatial_dims
        self.se_reduce = nn.Conv2d(n_feature, n_hidden, 1, bias=True)
        self.se_expand = nn.Conv2d(n_hidden, n_feature, 1, bias=True)
        self.active_fn = active_fn()

    def forward(self, x):
        se_tensor = x.mean(self.spatial_dims, keepdim=True)
        se_tensor = self.se_expand(self.active_fn(self.se_reduce(se_tensor)))
        return torch.sigmoid(se_tensor) * x

    def __repr__(self):
        return '{}({}, {}, spatial_dims={}, active_fn={})'.format(
            self._get_name(), self.n_feature, self.n_hidden, self.spatial_dims,
            self.active_fn)


class ZeroInitBN(nn.BatchNorm2d):
    """BatchNorm with zero initialization."""

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)


class Nonlocal(nn.Module):
    """Lightweight Non-Local Module.

    See https://arxiv.org/abs/2004.01961
    """

    def __init__(self, n_feature, nl_c, nl_s, batch_norm_kwargs=None):
        super(Nonlocal, self).__init__()
        self.n_feature = n_feature
        self.nl_c = nl_c
        self.nl_s = nl_s
        self.depthwise_conv = nn.Conv2d(n_feature,
                                        n_feature,
                                        3,
                                        1, (3 - 1) // 2,
                                        groups=n_feature,
                                        bias=False)

        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        from utils.config import FLAGS
        if hasattr(FLAGS, 'nl_norm'):  # TODO: as param
            self.bn = get_nl_norm_fn(FLAGS.nl_norm)(n_feature,
                                                    **batch_norm_kwargs)
        else:
            self.bn = ZeroInitBN(n_feature, **batch_norm_kwargs)

    def forward(self, l):
        N, n_in, H, W = list(l.shape)
        reduced_HW = (H // self.nl_s) * (W // self.nl_s)
        l_reduced = l[:, :, ::self.nl_s, ::self.nl_s]
        theta, phi, g = l[:, :int(self.nl_c * n_in), :, :], l_reduced[:, :int(
            self.nl_c * n_in), :, :], l_reduced
        if (H * W) * reduced_HW * n_in * (1 + self.nl_c) < (
                H * W) * n_in**2 * self.nl_c + reduced_HW * n_in**2 * self.nl_c:
            f = torch.einsum('niab,nicd->nabcd', theta, phi)
            f = torch.einsum('nabcd,nicd->niab', f, g)
        else:
            f = torch.einsum('nihw,njhw->nij', phi, g)
            f = torch.einsum('nij,nihw->njhw', f, theta)
        f = f / H * W
        f = self.bn(self.depthwise_conv(f))
        return f + l

    def __repr__(self):
        return '{}({}, nl_c={}, nl_s={}'.format(self._get_name(),
                                                self.n_feature, self.nl_c,
                                                self.nl_s)


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 active_fn=None,
                 batch_norm_kwargs=None):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes, **batch_norm_kwargs), active_fn())


class ConvBNReLUFriendly(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 active_fn=None,
                 batch_norm_kwargs=None):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes, **batch_norm_kwargs), active_fn())


class InvertedResidualChannelsFused(nn.Module):
    """Speedup version of `InvertedResidualChannels` by fusing small kernels.

    NOTE: It may consume more GPU memory.
    Support `Squeeze-and-Excitation`.
    """

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 channels,
                 kernel_sizes,
                 expand,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 se_ratio=None,
                 nl_c=0,
                 nl_s=0):
        super(InvertedResidualChannelsFused, self).__init__()
        assert stride in [1, 2]
        assert len(channels) == len(kernel_sizes)

        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn
        self.se_ratio = se_ratio
        self.nl_c = nl_c
        self.nl_s = nl_s

        (self.expand_conv, self.depth_ops, self.project_conv, self.se_op,
         self.nl_op) = self._build(channels, kernel_sizes, expand, se_ratio,
                                   nl_c, nl_s)

    def _build(self, hidden_dims, kernel_sizes, expand, se_ratio, nl_c, nl_s):
        _batch_norm_kwargs = self.batch_norm_kwargs \
            if self.batch_norm_kwargs is not None else {}

        hidden_dim_total = sum(hidden_dims)
        if self.expand:
            # pw
            expand_conv = ConvBNReLU(self.input_dim,
                                     hidden_dim_total,
                                     kernel_size=1,
                                     batch_norm_kwargs=_batch_norm_kwargs,
                                     active_fn=self.active_fn)
        else:
            expand_conv = Identity()

        narrow_start = 0
        depth_ops = nn.ModuleList()
        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            if expand:
                layers.append(Narrow(1, narrow_start, hidden_dim))
                narrow_start += hidden_dim
            else:
                if hidden_dim != self.input_dim:
                    raise RuntimeError('uncomment this for search_first model')
                logging.warning(
                    'uncomment this for previous trained search_first model')
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim,
                           hidden_dim,
                           kernel_size=k,
                           stride=self.stride,
                           groups=hidden_dim,
                           batch_norm_kwargs=_batch_norm_kwargs,
                           active_fn=self.active_fn),
            ])
            depth_ops.append(nn.Sequential(*layers))
        project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim_total, self.output_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs))

        if expand and narrow_start != hidden_dim_total:
            raise ValueError('Part of expanded are not used')

        if se_ratio is not None and se_ratio > 0:
            se_op = SqueezeAndExcitation(hidden_dim_total,
                                         int(round(self.input_dim * se_ratio)),
                                         active_fn=self.active_fn)
        else:
            se_op = Identity()

        if nl_c > 0:
            nl_op = Nonlocal(self.output_dim,
                             nl_c,
                             nl_s,
                             batch_norm_kwargs=_batch_norm_kwargs)
        else:
            nl_op = Identity()
        return expand_conv, depth_ops, project_conv, se_op, nl_op

    def get_depthwise_bn(self):
        """Get `[module]` list of BN after depthwise convolution."""
        return list(self.get_named_depthwise_bn().values())

    def get_named_depthwise_bn(self, prefix=None):
        """Get `{name: module}` pairs of BN after depthwise convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.depth_ops):
            children = list(op.children())
            if self.expand:
                idx_op = 1
            else:
                raise RuntimeError('Not search_first')
            conv_bn_relu = children[idx_op]
            assert isinstance(conv_bn_relu, ConvBNReLU)
            conv_bn_relu = list(conv_bn_relu.children())
            _, bn, _ = conv_bn_relu
            assert isinstance(bn, nn.BatchNorm2d)
            name = 'depth_ops.{}.{}.1'.format(i, idx_op)
            name = add_prefix(name, prefix)
            res[name] = bn
        return res

    def forward(self, x):
        res = self.expand_conv(x)
        res = [op(res) for op in self.depth_ops]
        if len(res) != 1:
            res = torch.cat(res, dim=1)
        else:
            res = res[0]
        res = self.se_op(res)
        res = self.project_conv(res)
        res = self.nl_op(res)
        if self.use_res_connect:
            return x + res
        return res

    def __repr__(self):
        return ('{}({}, {}, channels={}, kernel_sizes={}, expand={}, stride={},'
                ' se_ratio={}, nl_s={}, nl_c={})').format(
                    self._get_name(), self.input_dim, self.output_dim,
                    self.channels, self.kernel_sizes, self.expand, self.stride,
                    self.se_ratio, self.nl_s, self.nl_c)


class InvertedResidual(nn.Module):
    """MobiletNetV2 building block."""

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 channels,
                 kernel_sizes,
                 expand,
                 active_fn=None,
                 batch_norm_kwargs=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert len(channels) == len(kernel_sizes)

        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn

        self.ops, self.pw_bn = self._build(channels, kernel_sizes, expand)

    def _build(self, hidden_dims, kernel_sizes, expand):
        _batch_norm_kwargs = self.batch_norm_kwargs \
            if self.batch_norm_kwargs is not None else {}

        narrow_start = 0
        ops = nn.ModuleList()
        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            if expand:
                # pw
                layers.append(
                    ConvBNReLU(self.input_dim,
                               hidden_dim,
                               kernel_size=1,
                               batch_norm_kwargs=_batch_norm_kwargs,
                               active_fn=self.active_fn))
            else:
                if hidden_dim != self.input_dim:
                    raise RuntimeError('uncomment this for search_first model')
                logging.warning(
                    'uncomment this for previous trained search_first model')
                # layers.append(Narrow(1, narrow_start, hidden_dim))
                narrow_start += hidden_dim
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim,
                           hidden_dim,
                           kernel_size=k,
                           stride=self.stride,
                           groups=hidden_dim,
                           batch_norm_kwargs=_batch_norm_kwargs,
                           active_fn=self.active_fn),
                # pw-linear
                nn.Conv2d(hidden_dim, self.output_dim, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup, **batch_norm_kwargs),
            ])
            ops.append(nn.Sequential(*layers))
        pw_bn = nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs)

        if not expand and narrow_start != self.input_dim:
            raise ValueError('Part of input are not used')
        return ops, pw_bn

    def get_depthwise_bn(self):
        """Get `[module]` list of BN after depthwise convolution."""
        return list(self.get_named_depthwise_bn().values())

    def get_named_depthwise_bn(self, prefix=None):
        """Get `{name: module}` pairs of BN after depthwise convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.ops):
            children = list(op.children())
            if self.expand:
                idx_op = 1
            else:
                idx_op = 0
            conv_bn_relu = children[idx_op]
            assert isinstance(conv_bn_relu, ConvBNReLU)
            conv_bn_relu = list(conv_bn_relu.children())
            _, bn, _ = conv_bn_relu
            assert isinstance(bn, nn.BatchNorm2d)
            name = 'ops.{}.{}.1'.format(i, idx_op)
            name = add_prefix(name, prefix)
            res[name] = bn
        return res

    def forward(self, x):
        tmp = sum(op(x) for op in self.ops)
        tmp = self.pw_bn(tmp)
        if self.use_res_connect:
            return x + tmp
        return tmp

    def __repr__(self):
        return ('{}({}, {}, channels={}, kernel_sizes={}, expand={},'
                ' stride={})').format(self._get_name(), self.input_dim,
                                      self.output_dim, self.channels,
                                      self.kernel_sizes, self.expand,
                                      self.stride)


class InvertedResidualFriendly(nn.Module):
    def __init__(self, inp, oup, kernel, stride, expand_ratio, batch_norm_kwargs=None):
        super(InvertedResidualFriendly, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        _batch_norm_kwargs = batch_norm_kwargs if batch_norm_kwargs is not None else {}

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim, **_batch_norm_kwargs)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim//2, hidden_dim//2, (1, kernel), stride, (0, kernel//2), groups=hidden_dim//2, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim//2, **_batch_norm_kwargs)
        self.conv2_v = nn.Conv2d(hidden_dim//2, hidden_dim//2, (kernel, 1), stride, (kernel//2,0), groups=hidden_dim//2, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim//2, **_batch_norm_kwargs)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup, **_batch_norm_kwargs)

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

class InvertedResidualFriendly2(nn.Module):
    def __init__(self, inp, oup, kernel, stride, expand_ratio, batch_norm_kwargs=None):
        super(InvertedResidualFriendly2, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        _batch_norm_kwargs = batch_norm_kwargs \
            if batch_norm_kwargs is not None else {}

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim, **_batch_norm_kwargs)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim, hidden_dim, (1, kernel), stride, (0, kernel//2), groups=hidden_dim, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim, **_batch_norm_kwargs)
        self.conv2_v = nn.Conv2d(hidden_dim, hidden_dim, (kernel, 1), stride, (kernel//2,0), groups=hidden_dim, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim, **_batch_norm_kwargs)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(2*hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup, **_batch_norm_kwargs)

    def forward(self, x):
        if self.expand_ratio == 1:
            out = x
        else:
            out = self.r1(self.bn1(self.conv1(x)))

        out1 = self.r2(self.bn2_h(self.conv2_h(out)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.identity:
            return x + out
        else:
            return out


def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'nn.ReLU6': functools.partial(nn.ReLU6, inplace=True),
        'nn.ReLU': functools.partial(nn.ReLU, inplace=True),
        'nn.Swish': Swish,
        'nn.HSwish': HSwish,
    }[name]
    return active_fn


def get_nl_norm_fn(name):
    active_fn = {
        'nn.BatchNorm':
            ZeroInitBN,
        'nn.InstanceNorm':
            functools.partial(nn.InstanceNorm2d,
                              affine=True,
                              track_running_stats=True),
    }[name]
    return active_fn


def get_block(name):
    """Select building block."""
    return {
        'InvertedResidual': InvertedResidual,
        'InvertedResidualFriendly': InvertedResidualFriendly,
        'InvertedResidualFriendly2': InvertedResidualFriendly2,
        'MobileBottleneck': MobileBottleneck,
        'MobileBottleneckFriendly2': MobileBottleneckFriendly2
    }[name]


def init_weights_slimmable(m):
    """Slimmable network style initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            if isinstance(m, ZeroInitBN):
                nn.init.zeros_(m.weight)
            else:
                nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)


def init_weights_mnas(m):
    """MnasNet style initialization."""
    if isinstance(m, nn.Conv2d):
        if m.groups == m.in_channels:  # depthwise conv
            fan_out = m.weight[0][0].numel()
        else:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(fan_out)
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            if isinstance(m, ZeroInitBN):
                nn.init.zeros_(m.weight)
            else:
                nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.affine:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        init_range = 1.0 / np.sqrt(fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        if m.bias is not None:
            nn.init.constant_(m.bias,0)

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

class MobileBottleneckFriendly(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneckFriendly, self).__init__()
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

class MobileBottleneckFriendly2(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneckFriendly2, self).__init__()
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




