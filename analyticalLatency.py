import os
import sys
import time
import math
import math 

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np 
from tqdm import tqdm

def gemmCycles(dimension_rows, dimension_cols, ifmap_h, ifmap_w, filt_h, filt_w,
            num_channels, strides, num_filt):
        H = ifmap_h
        W = ifmap_w
        C = num_channels
        M = num_filt
        R = filt_h
        S = filt_w
        Stride = strides
        arrX = dimension_rows
        arrY = dimension_cols

        E = (H - R + Stride)/Stride
        F = (W - S + Stride)/Stride
    
        ## Reduce to Mat mul of A x B and  B X C
        numInput = E * F
        numTime  = R * S * C
        numFilter= M

        ## Utilization and Max Bandwidth
        numFolds = 0 ## To compute avearge
        weightedUtili = 0.0
        avgUtilization = 0.0
        maxBandwidth = 0.0

        cycles = 0
        cycles = (numInput//arrX) * (numFilter//arrY) * (numTime + arrX + arrY - 1)
        numFolds = (numInput//arrX) * (numFilter//arrY)
        weightedUtili = (numInput//arrX) * (numFilter//arrY) * 1.0
        
        if numInput % arrX > 0:
            cycles = cycles + (numFilter//arrY) * (numTime + (numInput % arrX) + arrY - 1)
            numFolds += (numFilter//arrY)
            weightedUtili += (numFilter//arrY) * ((numInput % arrX)*arrY/(arrX * arrY))
        
        if numFilter % arrY > 0:
            cycles = cycles + (numInput//arrX) * (numTime + arrX + (numFilter % arrY) - 1)
            numFolds += (numInput//arrX)
            weightedUtili += (numInput//arrX) * (arrX*(numFilter % arrY)/(arrX * arrY))

        if numInput % arrX > 0 and numFilter % arrY > 0:
            cycles = cycles + (numTime + (numInput % arrX) + (numFilter % arrY) - 1)
            numFolds += 1
            weightedUtili += 1 * ((numInput % arrX)*(numFilter % arrY)/(arrX * arrY))

        avgUtilization = weightedUtili/numFolds
        return math.ceil(cycles), avgUtilization

class Latency:
    def __init__(self):
        self.time = 0
        self.pointwiseConv = 0
        self.depthwiseConv = 0
        self.otherConv = 0

class Utilization:
    def __init__(self):
        self.layerwiseUtilization = []
        self.avgUtilization = 0.0
        self.layercount = 0
        self.sum = 0.0

    def update(self, utili):
        self.layerwiseUtilization.append(utili)
        self.sum += utili
        self.layercount += 1
        self.avgUtilization = self.sum/self.layercount

    def updateFuSe(self, utili):
        self.layerwiseUtilization.append(utili)
        self.sum += 0.5 * utili
        self.layercount += 0.5
        self.avgUtilization = self.sum/self.layercount

class ForwardHook:
    def __init__(self, arraySizeX, arraySizeY, hardware):
        self.latency = Latency()
        self.utilize = Utilization()
        self.arraySizeX = arraySizeX
        self.arraySizeY = arraySizeY
        assert hardware == 'FuSe'or hardware == 'Systolic'
        self.hardware = hardware

    def __call__(self, module, module_in, module_out):
        if isinstance(module, nn.Conv2d):
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
            
            # Groups == 1. Normal Convolution. Maps as GEMM op on Systolic and FuSe.
            if g == 1:
                t, u = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=inC,strides=s_h, num_filt=outC)
                # print('Group=1 ', inDim_h, inDim_w, k_h, k_w, inC, outC, t)
                self.utilize.update(u)
                if k_h == 1 and k_w == 1:
                    self.latency.pointwiseConv += t
                else:
                    self.latency.otherConv += t
            
            # Groups != 1. Therefore its a Depthwise Convolution. (PitFall)
            else:
                # If Systolic Hardware: Do Poor Utiliation GEMM. With 1 channel and 1 filter.
                if self.hardware == 'Systolic':
                    t, u = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=1,strides=s_h, num_filt=1)
                    t = t*outC
                    self.utilize.update(u)
                    self.latency.depthwiseConv += t
                
                elif self.hardware == 'FuSe':
                    # On FuSe, If its spatial DW conv, do poor utilization GEMM
                    # Else with FuSe networks, do FuseConv
                    if k_h != 1 and k_w != 1:
                        t, u = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=1,strides=s_h, num_filt=1)
                        t = t*outC
                        self.utilize.update(u)
                        self.latency.depthwiseConv += t
                    # Case: 1 x K kernel
                    elif k_h == 1:
                        # No of 1D convs performed.
                        # No of Folds in X direction. 
                        # No of Folds in Y direction.
                        # Time for one Fold  (Compute happens only in Y direction.0 
                        num1Dconv = inDim_h * outC 
                        numFoldsX = num1Dconv/self.arraySizeX
                        numFoldsY = inDim_w/self.arraySizeY
                        oneFoldTime = self.arraySizeY + k_w

                        t = math.ceil((math.ceil(numFoldsX)/s_w)*(oneFoldTime*math.ceil(numFoldsY)))
                        self.latency.depthwiseConv += t

                        ## Utilization
                        outDim_h = inDim_h/s_h
                        outDim_w = (inDim_w-k_w+1)/s_w

                        if outDim_h >= self.arraySizeX:
                            if outDim_w >= self.arraySizeY:
                                u = 1.0
                            else: 
                                u = outDim_w/self.arraySizeY
                        else:
                            if outDim_w >= self.arraySizeY:
                                u = outDim_h/self.arraySizeX
                            else: 
                                u = (outDim_h/self.arraySizeX)*(outDim_w/self.arraySizeY)
                        
                        self.utilize.updateFuSe(u)
                    # Case : K x 1 kernel
                    elif k_w == 1:
                        num1Dconv = inDim_w * outC
                        numFoldsX = num1Dconv/self.arraySizeY
                        numFoldsY = inDim_h/self.arraySizeX
                        oneFoldTime = self.arraySizeX + k_h
                        t = math.ceil((math.ceil(numFoldsX)/s_h)*(oneFoldTime*math.ceil(numFoldsY)))
                        self.latency.depthwiseConv += t

                        ## Utilization
                        outDim_h = (inDim_h-k_h+1)/s_h
                        outDim_w = (inDim_w)/s_w

                        if outDim_w >= self.arraySizeX:
                            if outDim_h >= self.arraySizeY:
                                u = 1.0
                            else: 
                                u = outDim_h/self.arraySizeY
                        else:
                            if outDim_h >= self.arraySizeY:
                                u = outDim_w/self.arraySizeX
                            else: 
                                u = (outDim_w/self.arraySizeX)*(outDim_h/self.arraySizeY)
                        self.utilize.updateFuSe(u)

            self.latency.time += t
        
        elif isinstance(module, nn.Linear):
            inT = module_in[0]
            inDim_h, inDim_w = (inT.shape[0], inT.shape[1])
            assert inDim_h == 1
            inC = module.in_features
            outC = module.out_features
            t, u = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=1, ifmap_w=1,
                                filt_h=1, filt_w=1,
                                num_channels=inC,strides=1, num_filt=outC)
            self.utilize.update(u)
            self.latency.otherConv += t
            self.latency.time += t
    
    def clear(self):
        self.latency = Latency()
        self.utilize = Utilization()

def getModelLatency(model, x, arraySizeX=8, arraySizeY=8, hardware='Systolic'):    
    hookfn = ForwardHook(arraySizeX, arraySizeY, hardware)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hookfn)
        elif isinstance(layer, nn.Linear):
            layer.register_forward_hook(hookfn)
    model(x)
    latency = hookfn.latency.time
    hookfn.clear()
    return latency

def getModelUtili(model, x, arraySizeX=8, arraySizeY=8, hardware='Systolic'):    
    hookfn = ForwardHook(arraySizeX, arraySizeY, hardware)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hookfn)
        elif isinstance(layer, nn.Linear):
            layer.register_forward_hook(hookfn)
    model(x)
    layerUtili = hookfn.utilize.layerwiseUtilization
    avgUtili = hookfn.utilize.avgUtilization
    hookfn.clear()
    return layerUtili, avgUtili

# def getModelLatencyBreakdown(model, x, mode='analytical', arraySize=8):    
#     hookfn = ForwardHook(arraySize, mode)
#     for layer in model.modules():
#         if isinstance(layer, nn.Conv2d):
#             layer.register_forward_hook(hookfn)
#         elif isinstance(layer, nn.Linear):
#             layer.register_forward_hook(hookfn)
#     model(x)
#     totalLatency = hookfn.time
#     otherConvLatency = hookfn.otherConv
#     pointConvLatency = hookfn.pointwiseConv
#     depthConvLatency = hookfn.depthwiseConv
#     linearLatency = hookfn.linear
#     hookfn.clear()
#     return otherConvLatency, pointConvLatency, depthConvLatency, linearLatency

x = torch.randn([1,3,224,224])
hardware = 'Systolic' ## or 'FuSe'
arraySizeX = 8
arraySizeY = 8
from models import *
supernet = [MobileNetV1(1000), MobileNetV2(1000), MnasNet(1000), MobileNetV3('small', 1000), MobileNetV3('large', 1000)]
supernetf1 = [MobileNetV1Friendly(1000), MobileNetV2Friendly(1000), MnasNetFriendly(1000), MobileNetV3Friendly('small', 1000), MobileNetV3Friendly('large', 1000)]
for net in supernet:
    # print( getModelLatency(net, x, arraySizeX, arraySizeY, hardware))
    a, b = getModelUtili(net, x, arraySizeX, arraySizeY, hardware)
    print(b)
print("Friendly")
for net in supernetf1:
    # print( getModelLatency(net, x, arraySizeX, arraySizeY, hardware))
    a, b = getModelUtili(net, x, arraySizeX, arraySizeY, 'FuSe')
    print(b)
