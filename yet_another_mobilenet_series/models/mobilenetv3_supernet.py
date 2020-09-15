"""Supernet builder."""
import numbers
from torch import nn
import functools
from models.mobilenet_base import _make_divisible
from models.mobilenet_base import ConvBNReLU
from models.mobilenet_base import get_active_fn
from models.mobilenet_base import get_block
import torch.nn.functional as F

__all__ = ['MobileNetV3']

class Hswish_new(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish_new, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

def get_block_wrapper(block_str):
    """ Wrapper for MobileNetV3 Block."""

    class MobileBottleneck(get_block(block_str)):
        def __init__(self,
                     inp,
                     oup,
                     kernel,
                     stride,
                     exp,
                     se=False,
                     nl='RE',
                     batch_norm_kwargs=None):

            #def _expand_ratio_to_hiddens(expand_ratio):
            #    if isinstance(expand_ratio, list):
            #        expand = True
            #    elif isinstance(expand_ratio, numbers.Number):
            #        expand = expand_ratio != 1
            #        expand_ratio = [expand_ratio for _ in kernel_sizes]
            #    else:
            #        raise ValueError(
            #            'Unknown expand_ratio type: {}'.format(expand_ratio))
            #    hidden_dims = [int(round(inp * e)) for e in expand_ratio]
            #    return hidden_dims, expand

            #hidden_dims, expand = _expand_ratio_to_hiddens(expand_ratio)
            super(MobileBottleneck,
                  self).__init__(inp,
                                 oup,
                                 kernel,
                                 stride,
                                 exp,
                                 se,
                                 nl,
                                 batch_norm_kwargs
                                )
            self.exp = exp

    return MobileBottleneck

def get_block_wrapper_friendly(block_str):
    """ Wrapper for MobileNetV3 Block."""

    class MobileBottleneckFriendly(get_block(block_str)):
        def __init__(self,
                     inp,
                     oup,
                     kernel,
                     stride,
                     exp,
                     se=False,
                     nl='RE',
                     batch_norm_kwargs=None):

            #def _expand_ratio_to_hiddens(expand_ratio):
            #    if isinstance(expand_ratio, list):
            #        expand = True
            #    elif isinstance(expand_ratio, numbers.Number):
            #        expand = expand_ratio != 1
            #        expand_ratio = [expand_ratio for _ in kernel_sizes]
            #    else:
            #        raise ValueError(
            #            'Unknown expand_ratio type: {}'.format(expand_ratio))
            #    hidden_dims = [int(round(inp * e)) for e in expand_ratio]
            #    return hidden_dims, expand

            #hidden_dims, expand = _expand_ratio_to_hiddens(expand_ratio)
            super(MobileBottleneckFriendly,
                  self).__init__(inp,
                                 oup,
                                 kernel,
                                 stride,
                                 exp,
                                 se,
                                 nl,
                                 batch_norm_kwargs
                                )
            self.exp = exp

    return MobileBottleneckFriendly

def get_block_wrapper_friendly2(block_str):
    """ Wrapper for MobileNetV3 Block."""

    class MobileBottleneckFriendly2(get_block(block_str)):
        def __init__(self,
                     inp,
                     oup,
                     kernel,
                     stride,
                     exp,
                     se=False,
                     nl='RE',
                     batch_norm_kwargs=None):

            #def _expand_ratio_to_hiddens(expand_ratio):
            #    if isinstance(expand_ratio, list):
            #        expand = True
            #    elif isinstance(expand_ratio, numbers.Number):
            #        expand = expand_ratio != 1
            #        expand_ratio = [expand_ratio for _ in kernel_sizes]
            #    else:
            #        raise ValueError(
            #            'Unknown expand_ratio type: {}'.format(expand_ratio))
            #    hidden_dims = [int(round(inp * e)) for e in expand_ratio]
            #    return hidden_dims, expand

            #hidden_dims, expand = _expand_ratio_to_hiddens(expand_ratio)
            super(MobileBottleneckFriendly2,
                  self).__init__(inp,
                                 oup,
                                 kernel,
                                 stride,
                                 exp,
                                 se,
                                 nl,
                                 batch_norm_kwargs
                                )
            self.exp = exp

    return MobileBottleneckFriendly2

class MobileNetV3(nn.Module):
    """MobileNetV2-like network."""

    def __init__(self,
                 num_classes=1000,
                 input_size=224,
                 input_channel=16,
                 last_channel=1280,
                 width_mult=1.0,
                 inverted_bottleneck_setting_small=None,
                 inverted_bottleneck_setting_large=None,
                 dropout_ratio=0.2,
                 batch_norm_momentum=0.1,
                 batch_norm_epsilon=1e-5,
                 active_fn='nn.ReLU6',
                 mode = 'large',
                 block='MobileBottleneck',
                 blockFriendly='MobileBottleneckFriendly2',
                 round_nearest=8):
        """Build the network.

        Args:
            num_classes (int): Number of classes
            input_size (int): Input resolution.
            input_channel (int): Number of channels for stem convolution.
            last_channel (int): Number of channels for stem convolution.
            width_mult (float): Width multiplier - adjusts number of channels in
                each layer by this amount
            inverted_residual_setting (list): A list of
                [expand ratio, output channel, num repeat,
                stride of first block, A list of kernel sizes].
            dropout_ratio (float): Dropout ratio for linear classifier.
            batch_norm_momentum (float): Momentum for batch normalization.
            batch_norm_epsilon (float): Epsilon for batch normalization.
            active_fn (str): Specify which activation function to use.
            block (str): Specify which MobilenetV2 block implementation to use.
            round_nearest (int): Round the number of channels in each layer to
                be a multiple of this number Set to 1 to turn off rounding.
        """
        super(MobileNetV3, self).__init__()
        batch_norm_kwargs = {
            'momentum': batch_norm_momentum,
            'eps': batch_norm_epsilon
        }

        self.input_channel = input_channel
        self.last_channel = last_channel
        self.width_mult = width_mult
        self.round_nearest = round_nearest
        self.inverted_bottleneck_setting_large = inverted_bottleneck_setting_large
        self.inverted_bottleneck_setting_small = inverted_bottleneck_setting_small
        self.active_fn = active_fn
        self.block = block
        self.blockFriendly = blockFriendly
        self.mode = mode
        
        if self.mode == 'large':
            self.inverted_bottleneck_setting = inverted_bottleneck_setting_large
        else:
            self.inverted_bottleneck_setting = inverted_bottleneck_setting_small


        if input_size % 32 != 0:
            raise ValueError('Input size must divide 32')
        active_fn = get_active_fn(active_fn)
        #block = get_block_wrapper(block)
        blockFriendly = get_block_wrapper_friendly2(blockFriendly)
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult),
                                       round_nearest)
        features = [
            ConvBNReLU(3,
                       input_channel,
                       stride=2,
                       batch_norm_kwargs=batch_norm_kwargs,
                       active_fn=Hswish_new)
        ]
        # building inverted residual blocks
        for k, exp, c, se, nl, s in self.inverted_bottleneck_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            exp_channel = _make_divisible(exp * width_mult, round_nearest)
            features.append(blockFriendly(input_channel, output_channel, k, s, exp_channel, se, nl, batch_norm_kwargs=batch_norm_kwargs))
            input_channel = output_channel

        if self.mode == 'large':
            last_conv = _make_divisible(960 * width_mult, round_nearest)
        else:
            last_conv = _make_divisible(576 * width_mult, round_nearest)
        features.append(
            ConvBNReLU(input_channel,
                       last_conv,
                       kernel_size=1,
                       batch_norm_kwargs=batch_norm_kwargs,
                       active_fn=functools.partial(nn.ReLU, inplace=True)))
        features.append(nn.AdaptiveAvgPool2d(1))
        #avg_pool_size = input_size // 32
        #features.append(nn.AvgPool2d(avg_pool_size))
        features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        features.append(Hswish_new())
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(3).squeeze(2)
        x = self.classifier(x)
        return x

Model = MobileNetV3
#Model = MobileNetV2
