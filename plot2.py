import torch
import time
from models import *
from utils import *

class ForwardHook:
    def __init__(self, arraySize, mode):
        self.time = 0
        self.pointwiseConv = 0
        self.pointwiseConvList = []
        self.depthwiseConvList = []
        self.depthwiseConv = 0
        self.otherConv = 0
        self.arraySize = arraySize
        if mode == 'analytical':
            self.latencyFn = gemmCycles
        else:
            self.latencyFn = sram_traffic
    def __call__(self, module, module_in, module_out):
        inT = module_in[0]
        inDim_h, inDim_w = (inT.shape[2], inT.shape[3])
        inC = module.in_channels
        outC = module.out_channels
        k_h, k_w = module.kernel_size
        s_h, s_w = module.stride
        p_h, p_w = module.padding
        g = module.groups
        inDim_h = inDim_h + 2*p_h
        inDim_w = inDim_w + 2*p_w
        if g == 1:
            t = self.latencyFn(dimension_rows=self.arraySize, dimension_cols=self.arraySize, 
                            ifmap_h=inDim_h, ifmap_w=inDim_w,
                            filt_h=k_h, filt_w=k_w,
                            num_channels=inC,strides=s_h, num_filt=outC)
            # print('Group=1 ', inDim_h, inDim_w, k_h, k_w, inC, outC, t)
            t = int(t)
            if k_h == 1 and k_w == 1:
                self.pointwiseConv += t
                self.pointwiseConvList.append(t)
            else:
                self.otherConv += t
        else:
            if k_h == 1:
                num1Dconv = inDim_h * outC 
                numFolds = num1Dconv/self.arraySize
                oneFoldTime = self.arraySize + k_w
                num1DconvRow = inDim_h/self.arraySize
                time = (math.ceil(numFolds)/s_w)*(oneFoldTime*math.ceil(num1DconvRow))
                time = math.ceil(time)
                t = time
                self.depthwiseConv += t
                self.depthwiseConvList.append(t)
            elif k_w ==1 :
                num1Dconv = inDim_w * outC
                numFolds = num1Dconv/self.arraySize
                oneFoldTime = self.arraySize + k_h
                num1DconvRow = inDim_w/self.arraySize
                time = (math.ceil(numFolds)/s_w)*(oneFoldTime*math.ceil(num1DconvRow))
                time = math.ceil(time)
                t = time
                self.depthwiseConv += t
                self.depthwiseConvList.append(t)
            else:
                t = self.latencyFn(dimension_rows=self.arraySize, dimension_cols=self.arraySize, 
                            ifmap_h=inDim_h, ifmap_w=inDim_w,
                            filt_h=k_h, filt_w=k_w,
                            num_channels=1,strides=s_h, num_filt=1)
                t = int(t)
                t = t*outC
                self.depthwiseConv += t
                self.depthwiseConvList.append(t)
            # print('Group > 1 ', inDim_h, inDim_w, k_h, k_w, inC, outC, t)

        self.time += t
    
    def clear(self):
        self.time = 0
        self.pointwiseConv = 0
        self.depthwiseConv = 0
        self.otherConv = 0
        self.depthwiseConvList = []
        self.pointwiseConvList = []

def getModelLatency(model, x, mode='analytical', arraySize=8):    
    hookfn = ForwardHook(arraySize, mode)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hookfn)
    model(x)
    latency = hookfn.time
    print(hookfn.pointwiseConvList)
    print(hookfn.depthwiseConvList)
    hookfn.clear()
    return latency

x = torch.randn([1,3,224,224])
mode = 'scale-sim'
mode = 'analytical'
arraySize = 64

net1 = MobileNetV2(1000)
net2 = MobileNetV2Friendly(1000)
# net3 = MobileNetV2Friendly2(1000)
lat1 = getModelLatency(net1, x, mode, arraySize)
lat2 = getModelLatency(net2, x, mode, arraySize)
# lat3 = getModelLatency(net3, x, mode, arraySize)
print(lat1, lat2, lat1/lat2)