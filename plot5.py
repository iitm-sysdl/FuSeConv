import torch
import time
from models import *
from utils import *
# from thop import profile
from pthflops import count_ops

import mobilenetv3fusedHybrid as mv3hybrid
import mobilenetv2fusedHybrid as mv2hybrid
import mobilenetv1hybrid as mv1hybrid
import mnasnethybrid as mnashybrid

supernet = [MobileNetV1(1000), MobileNetV2(1000), MnasNet(1000), MobileNetV3('small', 1000), MobileNetV3('large', 1000)]
supernetf1 = [MobileNetV1Friendly(1000), MobileNetV2Friendly(1000), MnasNetFriendly(1000), MobileNetV3Friendly('small', 1000), MobileNetV3Friendly('large', 1000)]
supernetf2 = [MobileNetV1Friendly2(1000), MobileNetV2Friendly2(1000), MnasNetFriendly2(1000), MobileNetV3Friendly2('small', 1000), MobileNetV3Friendly2('large', 1000)]
superneth1 = [mv1hybrid.MobileNetV1Friendly(1000), mv2hybrid.MobileNetV2Friendly(1000), mnashybrid.MnasNetFriendly(1000), mv3hybrid.MobileNetV3Friendly('small', 1000), mv3hybrid.MobileNetV3Friendly('large', 1000)]
superneth2 = [mv1hybrid.MobileNetV1Friendly2(1000), mv2hybrid.MobileNetV2Friendly2(1000), mnashybrid.MnasNetFriendly2(1000), mv3hybrid.MobileNetV3Friendly2('small', 1000), mv3hybrid.MobileNetV3Friendly2('large', 1000)]


## utils.py: [578.875904, 320.236288, 330.189952, 64.333022, 232.162038] [567.285248, 306.425344, 309.040768, 58.575326, 218.661558] [1116.43648, 438.350592, 447.401088, 80.507516, 313.81694] [571.048448, 311.656192, 316.98112, 61.00259, 224.193462] [757.402112, 368.405248, 367.483264, 70.711544, 256.355588]
## pthflops: [588.96128, 314.34688, 325.68352, 66.944468, 238.725476] [573.507072, 300.535936, 304.534336, 61.186772, 225.224996] [1122.658304, 430.15936, 440.950336, 84.053224, 322.877] [578.524672, 305.766784, 312.474688, 63.614036, 230.7569] [764.878336, 361.085824, 361.894912, 73.830384, 264.007608]
x = torch.randn([1,3,224,224])
flopsb = []
flopsf1 = []
flopsf2 = []
flopsh1 = []
flopsh2 = []
paramsb = []
paramsf1 = []
paramsf2 = []
paramsh1 = []
paramsh2 = []

def countParams(model):
    return sum(p.numel() for p in model.parameters())

for net in supernet:
    paramsb.append(countParams(net)/1e6)
for net in supernetf1:
    paramsf1.append(countParams(net)/1e6)
for net in supernetf2:
    paramsf2.append(countParams(net)/1e6)
for net in superneth1:
    paramsh1.append(countParams(net)/1e6)
for net in superneth2:
    paramsh2.append(countParams(net)/1e6)

print(paramsb, paramsf1, paramsf2, paramsh1, paramsh2)
#exit()

for net in supernet:
    f, data = count_ops(net, x)
    flopsb.append(f/1e6)
for net in supernetf1:
    f, data = count_ops(net, x)
    flopsf1.append(f/1e6)
for net in supernetf2:
    f, data = count_ops(net, x)
    flopsf2.append(f/1e6)
for net in superneth1:
    f, data = count_ops(net, x)
    flopsh1.append(f/1e6)
for net in superneth2:
    f, data = count_ops(net, x)
    flopsh2.append(f/1e6)
print(flopsb, flopsf1, flopsf2, flopsh1,flopsh2)
