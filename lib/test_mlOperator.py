#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:21:48 2018

@author: zhay
"""

#Test ml Operator class
import imp
#import os
import scipy as sp
#import pybrain as pb
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mlModel import DummyMLOperator
import deep_util


#%%
datafile='/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_30/RUCKMAN_RANCH_30_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_31/RUCKMAN_RANCH_31_merge_cut_100hz_learning.csv'

#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_32/RUCKMAN_RANCH_32_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_35/RUCKMAN_RANCH_35_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_34/RUCKMAN_RANCH_34_merge_cut_100hz_learning.csv'
#
data=deep_util.loadDEEPData(datafile,savePickle=True)


#%%  

channelList=data.columns
op=DummyMLOperator(dt=0.01)
ntStep=30
op.initialize(ntStep=ntStep,nchannel=2,channelList=['01_GLOBAL_GENERIC_SURFACE-SRPM','01_GLOBAL_PASON_TTS-STRQ'])
ntTot=data.shape[0]
nwin=ntTot//ntStep

target = np.zeros(data.shape[0])
#simulate streaming environment 
for iwin in range(nwin):
    ibegin=iwin*ntStep
    iend = ibegin+ntStep
    df = data[ibegin:iend] 
    dataList=op.getData(df)
    target[ibegin:iend]=op.process(dataList)


#%% test sticslip operator
    
import deep_util
import matplotlib.pyplot as plt
from stickslipDetector import StickSlipDetector
import pdb
# load data
datafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv'    #
data = deep_util.loadDEEPData(datafile, savePickle=True)

depthMin=10000/3.28;
depthMax=11600/3.28;

if('01_Global-Generic_Surface-HOLEDEPTH' in data.columns):
    holedepthName='01_Global-Generic_Surface-HOLEDEPTH'
    bitdepthName='01_Global-Generic_Surface-BIT_DEPTH'
else:
    holedepthName='01_GLOBAL_GENERIC_SURFACE-HOLEDEPTH'
    bitdepthName='01_GLOBAL_GENERIC_SURFACE-BIT_DEPTH'


mask= (data[holedepthName]>depthMin) & (data[holedepthName]< depthMax)

deepDF = data.loc[mask]


# set model parameters
model_name = '/home/zhay/DEEP/machineLearning/stickslipDetector/recurrent_model/stickslipDetector_conv_lstm16_merge.h5'
ntStep = 3000
ntOverlap = 0
batchSize=128

channelList=['01_GLOBAL_PASON_TTS-STRQ',
             '01_GLOBAL_GENERIC_SURFACE-SWOB',
             '01_GLOBAL_PASON_TTSV-DRPM',
             '01_GLOBAL_GENERIC_SURFACE-SRPM']
# create operator
op = StickSlipDetector(dt=0.01)

# op=DummyMLOperator(dt=0.01)

op.initialize(ntStep=ntStep,nchannel=3,rpmlimit=50,batchSize=batchSize,modelFile=model_name,isNormlize=True)
ntTot = deepDF.shape[0]
#deepArray=deepDF.
nwin = ntTot // ntStep

#target = np.zeros(deepDF.shape[0])
target=pd.Series(index=deepDF.index)
istart = 0
iend = ntStep*batchSize
# simulate streaming environment
nextLevel=1

#while iend < 30000:
while iend < deepDF.shape[0]:
#    pdb.set_trace()
    dataList = op.getData(deepDF[istart:iend],channelList=channelList)
    target[istart:iend] = op.process(dataList)
    istart += ntStep*batchSize
    iend = istart + ntStep*batchSize
    perc = iend /ntTot * 100
    if perc > nextLevel:
        print('%2.0f percent finished' % perc)
        nextLevel+=1


plt.figure()
ax1=plt.subplot(211)    

#plt.plot(emsFlow,label='EMS flow rate')

plt.plot(deepDF['01_GLOBAL_PASON_TTS-STRQ'].div(1350),label='torque')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Torque')
plt.ylim(-10,50)

ax1b = ax1.twinx()
plt.plot(target,'-r',label='prediction')
plt.legend()
plt.ylim(-1,2)


ax2=plt.subplot(212,sharex=ax1)
plt.plot(deepDF['01_GLOBAL_PASON_TTSV-DRPM'],label='DRPM prediction')
plt.plot(deepDF['01_GLOBAL_GENERIC_SURFACE-SRPM'],label='SRPM')

plt.legend()
plt.xlabel('Time')
plt.ylabel('DRPM')
plt.ylim(-100,200)

#%% test DrillingDataProcesser class
from DrillingDataProcessor import StickSlipProcessor

datafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv'    #
outfile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/test_prediction.csv'    #
cfgfile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/test_stickslip_proceesing.json'

op=StickSlipProcessor(dt=0.01,infile=datafile,outfile=outfile,cfgfile=cfgfile)
#op.loadOptions()
op.initialize()
dfOut=op.process()
#op.writeCSVData(dfOut,outfile)
#%%

plt.figure()
ax1=plt.subplot(211)    

#plt.plot(emsFlow,label='EMS flow rate')

plt.plot(dfOut['01_GLOBAL_PASON_TTS-STRQ'].div(1350),label='torque')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Torque')
plt.ylim(-10,50)

ax1b = ax1.twinx()
plt.plot(dfOut['SS_probability'],'-r',label='prediction')
plt.legend()
plt.ylim(-1,2)


ax2=plt.subplot(212,sharex=ax1)
plt.plot(dfOut['01_GLOBAL_PASON_TTSV-DRPM'],label='DRPM prediction')
plt.plot(dfOut['01_GLOBAL_GENERIC_SURFACE-SRPM'],label='SRPM')

plt.legend()
plt.xlabel('Time')
plt.ylabel('DRPM')
plt.ylim(-100,200)
