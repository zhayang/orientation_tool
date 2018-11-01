#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:11:19 2018
Time domain signal processing module
@author: zhay
"""


# import scipy as sp
#import pybrain as pb
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from scipy import signal
from scipy.fftpack import fft,ifft,fftfreq
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import blackman
#calculate normalized cross-correlation function between two time series of equal length



#%%pick time shift between two correlated signals

def getRicker(f,t,t0=0):
    pift = np.pi*f*(t-t0)
    wav = (1 - 2*pift**2)*np.exp(-pift**2)
    return wav



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


#%%calculate normalized cross-correlation function between two time series of equal length
def calcCCF(y1,y2,dt=0.01,fmin=0.05,fmax=40,normalize=True):
    fs=1/dt
    if fmax>=fs/2:
        fmax=fs/2-0.01
    
    b, a = signal.butter(4, np.array([fmin,fmax])/(fs/2), 'bandpass')

    y1_filt=filtfilt(b, a, y1)
    y2_filt=filtfilt(b, a, y2)

    y1_filt=signal.detrend(y1_filt)
    y2_filt=signal.detrend(y2_filt)
    
#    ccf_drpm=np.correlate(y1_filt,y2_filt,'full')
    ccf_raw=np.correlate(y1_filt,y2_filt,'full')
    
    if normalize:
        ccf=ccf_raw*(1/y1_filt.size/np.std(y2_filt)/np.std(y1_filt))  
    else:
        ccf=ccf_raw
        
    totallag=y1_filt.size
    timelag=(np.linspace(-totallag,totallag,totallag*2-1))*dt
#    maxCCFArray=np.max(ccf_drpm[maskLagRange])
#    print('optimal time shift is %2.2f sec'%timeShift)
    return ccf,timelag


#    Calculate normalized cross-correlation function between two time series of equal length, with stacking
def calcCCFStack(y1,y2,ntwin,dt=0.01,fmin=0.05,fmax=40,overlapPerc=0,normalize=True):
    fs=1/dt
    if fmax>=fs/2:
        fmax=fs/2-0.01
    
#    cut data into short segments   
    ntTot=y1.size
    nt_overlap= int(ntwin*overlapPerc)
    
#    filter
    
    b, a = signal.butter(4, np.array([fmin,fmax])/(fs/2), 'bandpass')

    y1_filt=filtfilt(b, a, y1)
    y2_filt=filtfilt(b, a, y2)

#    y1_filt=signal.detrend(y1_filt)
#    y2_filt=signal.detrend(y2_filt)
    
    ccf_stack=np.zeros((2*ntwin-1,))
    
    itStart=0
    itEnd=ntwin
    nwin_stack=0
    while itEnd<=ntTot:
        #    ccf_drpm=np.correlate(y1_filt,y2_filt,'full')
        y1_win=y1_filt[itStart:itEnd]
        y2_win=y2_filt[itStart:itEnd]
        ccf_raw=np.correlate(y1_win,y2_win,'full')
        if normalize:
            ccf_norm=ccf_raw*(1/ntwin/np.std(y1_win)/np.std(y2_win))
        else:
            ccf_norm=ccf_raw
        ccf_stack=ccf_stack+ccf_norm
        
        itStart=itStart+ntwin-nt_overlap
        itEnd=itStart+ntwin
        nwin_stack+=1
        
    totallag=ntwin
    timelag=(np.linspace(-totallag,totallag,totallag*2-1))*dt
    ccf=ccf_stack/nwin_stack
#    maxCCFArray=np.max(ccf_drpm[maskLagRange])
#    print('ntTot: ' + str(ntTot))
#    print('ntwin: ' + str(ntwin))
#    print('number of short window stacked: ' + str(nwin_stack))
    
    return ccf,timelag
