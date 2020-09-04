import torch
from models import *
from utils import *

net1 = ResNet50(1000)
net2 = ResNet50Friendly(1000)
net3 = ResNet50Friendly2(1000)
net4 = ResNet50Friendly3(1000)
net5 = ResNet50Friendly4(1000)
net6 = ResNet50Friendly5(1000)
net7 = ResNet50Friendly6(1000)

x = torch.randn([1,3,224,224])
latb = getModelLatency(net1, x)
lat1 = getModelLatency(net2, x)
lat2 = getModelLatency(net3, x)
lat3 = getModelLatency(net4, x)
lat4 = getModelLatency(net5, x)
lat5 = getModelLatency(net6, x)
lat6 = getModelLatency(net7, x)

print("ResNet SpeedUp")
print(latb/lat1, latb/lat2, latb/lat3, latb/lat4, latb/lat5, latb/lat6)

print("MobileNet Speedup")
net1 = MobileNetV3('large', 1000)
net2 = MobileNetV3Friendly('large', 1000)
x = torch.randn([1,3,224,224])
latb = getModelLatency(net1, x)
lat1 = getModelLatency(net2, x)
print(latb/lat1)