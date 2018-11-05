#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:50:48 2017

@author: zhay
"""



import scipy as sp
#import pybrain as pb
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from scipy import signal
from scipy.fftpack import fft,ifft,fftfreq
from scipy.signal import blackman
    

#%%running mean

def runningMean(dataIn,ntwin):
    ndim=len(dataIn.shape)
    nt=dataIn.shape[0]
    if ndim>1:
        ntr=dataIn.shape[1]
    else:
        ntr=1
       
    dataOut=np.zeros(dataIn.shape)
#   nt x nt matrix to store data
    for itr in range(0,ntr):
        for i in range(0,nt):
            offset=np.int(ntwin/2)
            idxArray=np.arange(i-offset,i+ntwin-offset)
            idxArray=idxArray[idxArray>0]
            idxArray=idxArray[idxArray<nt]
            if   ndim==1:
                dataOut[i]=np.mean(dataIn[idxArray])
            elif ndim==2:
                dataOut[i,itr]=np.mean(dataIn[idxArray])
    return dataOut