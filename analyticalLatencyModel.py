'''
FuSeConv: Fully Separable Convolutions for Fast Inference on Systolic Arrays
Authors: Surya Selvam, Vinod Ganesan, Pratyush Kumar
Email ID: selvams@purdue.edu, vinodg@cse.iitm.ac.in, pratyush@cse.iitm.ac.in
'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from models import *

def gemmCycles(dimension_rows, dimension_cols, ifmap_h, ifmap_w, filt_h, filt_w,
            num_channels, stride_h, stride_w, num_filt, batch_size = 1):
        
        N = batch_size
        H = ifmap_h
        W = ifmap_w
        C = num_channels
        M = num_filt
        R = filt_h
        S = filt_w
        StrideH = stride_h
        StrideW = stride_w
        arrX = dimension_rows
        arrY = dimension_cols

        E = (H - R + StrideH)//StrideH
        F = (W - S + StrideW)//StrideW

        ## Reduce to Mat mul of A x B and  B X C - Forward Pass (M x RSC with RSC x NEF to get M x NEF)
        ## Assuming M1: numFilter * numTime, M2: numTime * numInput
        numInput = N * E * F
        numTime  = R * S * C
        numFilter= M


        cycles = 0
        cycles = (numInput//arrX) * (numFilter//arrY) * (numTime + arrX + arrY - 1)
        
        if numInput % arrX > 0:
            cycles = cycles + (numFilter//arrY) * (numTime + (numInput % arrX) + arrY - 1)
        
        if numFilter % arrY > 0:
            cycles = cycles + (numInput//arrX) * (numTime + arrX + (numFilter % arrY) - 1)
            
        if numInput % arrX > 0 and numFilter % arrY > 0:
            cycles = cycles + (numTime + (numInput % arrX) + (numFilter % arrY) - 1)
     
        return cycles

def FuSeCycles(dimension_rows, dimension_cols, ifmap_h, ifmap_w, filt_h, filt_w,
            num_channels, stride_h, stride_w, num_filt, batch_size = 1):

        N = batch_size
        H = ifmap_h
        W = ifmap_w
        C = num_channels
        M = num_filt
        R = filt_h
        S = filt_w
        StrideH = stride_h
        StrideW = stride_w
        arrX = dimension_rows
        arrY = dimension_cols

        E = (H - R + StrideH)//StrideH
        F = (W - S + StrideW)//StrideW


        num1Dconv = N * H * C
        numFoldsX = num1Dconv/arrX
        numFoldsY = W/arrY
        oneFoldTime = arrY + S

        cycles = math.ceil((math.ceil(numFoldsX)/StrideW)*(oneFoldTime*math.ceil(numFoldsY)))

        return cycles

class Latency:
    def __init__(self):
        self.time = 0
        self.pointwiseConv = 0
        self.depthwiseConv = 0
        self.otherConv = 0

class ForwardHook:
    def __init__(self, arraySizeX, arraySizeY, hardware):
        self.latency = Latency()
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
            
            t = 0
            # Groups == 1. Normal Convolution. Maps as GEMM op on Systolic and FuSe.
            if g == 1:
                t = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=inC, stride_h=s_h, stride_w=s_w, num_filt=outC)
                if k_h == 1 and k_w == 1:
                    self.latency.pointwiseConv += t
                else:
                    self.latency.otherConv += t
            
            # Groups != 1. Therefore its a Depthwise Convolution.
            else:
                # If Systolic Hardware: Do Poor Utiliation GEMM. With 1 channel and 1 filter.
                if self.hardware == 'Systolic':
                    t = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=1,stride_h=s_h, stride_w=s_w, num_filt=1)
                    t = t*outC
                    self.latency.depthwiseConv += t
                
                elif self.hardware == 'FuSe':
                    # On FuSe, If its spatial KxK DW conv, do poor utilization GEMM
                    # Else with FuSe networks, do FuseConv
                    if k_h != 1 and k_w != 1:
                        t = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=1, stride_h=s_h, stride_w=s_w, num_filt=1)
                        t = t*outC
                        self.latency.depthwiseConv += t
                    # Case: 1 x K kernel. Assume 1 x K and Kx1 kernel occur symmetrica l.
                    elif k_h == 1:
                        t = FuSeCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=inDim_h, ifmap_w=inDim_w,
                                filt_h=k_h, filt_w=k_w,
                                num_channels=inC,stride_h=s_h, stride_w=s_w, num_filt=1)

                        self.latency.depthwiseConv += t
                    
                    elif k_w == 1:
                        t = FuSeCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=inDim_w, ifmap_w=inDim_h,
                                filt_h=k_w, filt_w=k_h,
                                num_channels=inC, stride_h=s_w, stride_w=s_h, num_filt=1)

                        self.latency.depthwiseConv += t
                        
            self.latency.time += t
        
        elif isinstance(module, nn.Linear):
            inT = module_in[0]
            inDim_h, inDim_w = (inT.shape[0], inT.shape[1])
            assert inDim_h == 1
            inC = module.in_features
            outC = module.out_features
            t = gemmCycles(dimension_rows=self.arraySizeX, dimension_cols=self.arraySizeY, 
                                ifmap_h=1, ifmap_w=1,
                                filt_h=1, filt_w=1,
                                num_channels=inC,stride_h=1, stride_w=1, num_filt=outC)

            self.latency.otherConv += t
            self.latency.time += t
    
    def clear(self):
        self.latency = Latency()

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

def test():
    num_classes = 1000
    baseline = [MnasNet(num_classes), MobileNetV1(num_classes), MobileNetV2(num_classes), MobileNetV3('small', num_classes), MobileNetV3('large', num_classes)]
    FuSeHalf = [MnasNetFuSeHalf(num_classes), MobileNetV1FuSeHalf(num_classes), MobileNetV2FuSeHalf(num_classes), MobileNetV3FuSeHalf('small', num_classes), MobileNetV3FuSeHalf('large', num_classes)]
    FuSeFull = [MnasNetFuSeFull(num_classes), MobileNetV1FuSeFull(num_classes), MobileNetV2FuSeFull(num_classes), MobileNetV3FuSeFull('small', num_classes), MobileNetV3FuSeFull('large', num_classes)]
    FuSeHalfHybrid = [MnasNetFuSeHalfHybrid(num_classes), MobileNetV1FuSeHalfHybrid(num_classes), MobileNetV2FuSeHalfHybrid(num_classes), MobileNetV3FuSeHalfHybrid('small', num_classes), MobileNetV3FuSeHalfHybrid('large', num_classes)]
    FuSeFullHybrid = [MnasNetFuSeFullHybrid(num_classes), MobileNetV1FuSeFullHybrid(num_classes), MobileNetV2FuSeFullHybrid(num_classes), MobileNetV3FuSeFullHybrid('small', num_classes), MobileNetV3FuSeFullHybrid('large', num_classes)]

    x = torch.rand([1,3,224,224])
    arrX = 64
    arrY = 64

    baselineLatency = []
    hardware = 'Systolic'
    for net in baseline:
        latency = getModelLatency(net, x, arrX, arrY, hardware)
        baselineLatency.append(latency)

    fuselatency = []
    hardware = 'FuSe'
    for net in FuSeHalf:
        latency = getModelLatency(net, x, arrX, arrY, hardware)
        fuselatency.append(latency)
    
    print(np.array(baselineLatency)/np.array(fuselatency))

if __name__ == '__main__':
    test()
