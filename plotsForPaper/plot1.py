import torch
import time
from models import *
from utils import *
from operator import truediv

network=['MobileNetV1', 'MobileNetV2', 'MnasNet', 'MobileNet-V3\nSmall', 'MobileNet-V3\nLarge']
baseline = []
v1 = []
v2 = []
h1 = []
h2 = []
x = torch.randn([1,3,224,224])
mode =  'analytical'
arraySize = 64

import mobilenetv3fusedHybrid as mv3hybrid
import mobilenetv2fusedHybrid as mv2hybrid
import mobilenetv1hybrid as mv1hybrid
import mnasnethybrid as mnashybrid


supernet = [MobileNetV1(1000), MobileNetV2(1000), MnasNet(1000), MobileNetV3('small', 1000), MobileNetV3('large', 1000)]
supernetf1 = [MobileNetV1Friendly(1000), MobileNetV2Friendly(1000), MnasNetFriendly(1000), MobileNetV3Friendly('small', 1000), MobileNetV3Friendly('large', 1000)]
supernetf2 = [MobileNetV1Friendly2(1000), MobileNetV2Friendly2(1000), MnasNetFriendly2(1000), MobileNetV3Friendly2('small', 1000), MobileNetV3Friendly2('large', 1000)]
superneth1 = [mv1hybrid.MobileNetV1Friendly(1000), mv2hybrid.MobileNetV2Friendly(1000), mnashybrid.MnasNetFriendly(1000), mv3hybrid.MobileNetV3Friendly('small', 1000), mv3hybrid.MobileNetV3Friendly('large', 1000)]
superneth2 = [mv1hybrid.MobileNetV1Friendly2(1000), mv2hybrid.MobileNetV2Friendly2(1000), mnashybrid.MnasNetFriendly2(1000), mv3hybrid.MobileNetV3Friendly2('small', 1000), mv3hybrid.MobileNetV3Friendly2('large', 1000)]

for i, net in enumerate(supernet):
    baseline.append(getModelLatency(supernet[i], x, mode, arraySize))
    v1.append(getModelLatency(supernetf1[i], x, mode, arraySize))
    v2.append(getModelLatency(supernetf2[i], x, mode, arraySize))
    h1.append(getModelLatency(superneth1[i], x, mode, arraySize))
    h2.append(getModelLatency(superneth2[i], x, mode, arraySize))

print(baseline)
print(v1)
print(v2)
print(h1)
print(h2)

def listDivide(a,b):
    return list(map(truediv, a, b))

print('Speedup')
print(listDivide(baseline,baseline))
print(listDivide(baseline,v1))
print(listDivide(baseline,v2))
print(listDivide(baseline,h1))
print(listDivide(baseline,h2))


# net1 = MobileNetV1(1000)
# net2 = MobileNetV1Friendly(1000)
# net3 = MobileNetV1Friendly2(1000)
# lat1 = getModelLatency(net1, x, mode, arraySize)
# lat2 = getModelLatency(net2, x, mode, arraySize)
# lat3 = getModelLatency(net3, x, mode, arraySize)
# speedup.append(lat1/lat2)
# speedup2.append(lat1/lat3)

# net1 = MobileNetV2(1000)
# net2 = MobileNetV2Friendly(1000)
# net3 = MobileNetV2Friendly2(1000)
# lat1 = getModelLatency(net1, x, mode, arraySize)
# lat2 = getModelLatency(net2, x, mode, arraySize)
# lat3 = getModelLatency(net3, x, mode, arraySize)
# speedup.append(lat1/lat2)
# speedup2.append(lat1/lat3)

# net1 = MnasNet(1000)
# net2 = MnasNetFriendly(1000)
# net3 = MnasNetFriendly2(1000)
# lat1 = getModelLatency(net1, x, mode, arraySize)
# lat2 = getModelLatency(net2, x, mode, arraySize)
# lat3 = getModelLatency(net3, x, mode, arraySize)
# speedup.append(lat1/lat2)
# speedup2.append(lat1/lat3)

# net1 = MobileNetV3('small', 1000)
# net2 = MobileNetV3Friendly('small', 1000)
# net3 = MobileNetV3Friendly2('small',1000)
# lat1 = getModelLatency(net1, x, mode, arraySize)
# lat2 = getModelLatency(net2, x, mode, arraySize)
# lat3 = getModelLatency(net3, x, mode, arraySize)
# speedup.append(lat1/lat2)
# speedup2.append(lat1/lat3)

# net1 = MobileNetV3('large', 1000)
# net2 = MobileNetV3Friendly('large', 1000)
# net3 = MobileNetV3Friendly2('large', 1000)
# lat1 = getModelLatency(net1, x, mode, arraySize)
# lat2 = getModelLatency(net2, x, mode, arraySize)
# lat3 = getModelLatency(net3, x, mode, arraySize)
# speedup.append(lat1/lat2)
# speedup2.append(lat1/lat3)

# print(speedup)
# print(speedup2)