import torch
import time
from models import *
from utils import *

network=['MobileNetV1', 'MobileNetV2', 'MnasNet', 'MobileNet-V3\nSmall', 'MobileNet-V3\nLarge']
speedup = []
speedup2 = []
x = torch.randn([1,3,224,224])
mode = 'scale-sim'
mode =  'analytical'
arraySize = 64

net1 = MobileNetV1(1000)
net2 = MobileNetV1Friendly(1000)
net3 = MobileNetV1Friendly2(1000)
lat1 = getModelLatency(net1, x, mode, arraySize)
lat2 = getModelLatency(net2, x, mode, arraySize)
lat3 = getModelLatency(net3, x, mode, arraySize)
speedup.append(lat1/lat2)
speedup2.append(lat1/lat3)

net1 = MobileNetV2(1000)
net2 = MobileNetV2Friendly(1000)
net3 = MobileNetV2Friendly2(1000)
lat1 = getModelLatency(net1, x, mode, arraySize)
lat2 = getModelLatency(net2, x, mode, arraySize)
lat3 = getModelLatency(net3, x, mode, arraySize)
speedup.append(lat1/lat2)
speedup2.append(lat1/lat3)

net1 = MnasNet(1000)
net2 = MnasNetFriendly(1000)
net3 = MnasNetFriendly2(1000)
lat1 = getModelLatency(net1, x, mode, arraySize)
lat2 = getModelLatency(net2, x, mode, arraySize)
lat3 = getModelLatency(net3, x, mode, arraySize)
speedup.append(lat1/lat2)
speedup2.append(lat1/lat3)

net1 = MobileNetV3('small', 1000)
net2 = MobileNetV3Friendly('small', 1000)
net3 = MobileNetV3Friendly2('small',1000)
lat1 = getModelLatency(net1, x, mode, arraySize)
lat2 = getModelLatency(net2, x, mode, arraySize)
lat3 = getModelLatency(net3, x, mode, arraySize)
speedup.append(lat1/lat2)
speedup2.append(lat1/lat3)

net1 = MobileNetV3('large', 1000)
net2 = MobileNetV3Friendly('large', 1000)
net3 = MobileNetV3Friendly2('large', 1000)
lat1 = getModelLatency(net1, x, mode, arraySize)
lat2 = getModelLatency(net2, x, mode, arraySize)
lat3 = getModelLatency(net3, x, mode, arraySize)
speedup.append(lat1/lat2)
speedup2.append(lat1/lat3)

print(speedup)
print(speedup2)