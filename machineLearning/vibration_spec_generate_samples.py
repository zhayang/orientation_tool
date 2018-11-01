#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 11, 2018
Generate training and dev samples of vibration data using spectrogram
@author: zhay
"""

#Try Identify severe stickslip using machien learning
#%% Script to load csv file to pandas dataframe
#%autoreload 2
import imp
#import os
import scipy as sp
#import pybrain as pb
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from spectral import calc_coherence,calc_spec
from scipy.signal import butter, lfilter, filtfilt

# %matplotlib inline
#%matplotlib notebook

#import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D,Conv2D, MaxPooling2D,MaxPooling1D
from keras.losses import binary_crossentropy, categorical_crossentropy

#import os
import pickle
from keras.models import load_model, save_model
#import h5py
#import numpy as np
from time import time
import DeepDataFrame
import keras_utility
imp.reload(keras_utility)
from keras_utility import precision, recall, binary_xentropy, fmeasure

from keras_utility import augmentData
#%autoreload 2
from StickSlipDetector import StickSlipDetector
import deep_util
import spectral
#%% load data
#datafile='/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv'
datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_30/RUCKMAN_RANCH_30_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_31/RUCKMAN_RANCH_31_merge_cut_100hz_learning.csv'

#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_32/RUCKMAN_RANCH_32_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_35/RUCKMAN_RANCH_35_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_34/RUCKMAN_RANCH_34_merge_cut_100hz_learning.csv'
#
data=deep_util.loadDEEPData(datafile,savePickle=True)



if('01_Global-Generic_Surface-HOLEDEPTH' in data.columns):
    device='01_Global-Generic_Surface'
    holedepthName=device+'-HOLEDEPTH'
    bitdepthName=device+'-BIT_DEPTH'
else:
    device='01_GLOBAL_GENERIC_SURFACE'
    holedepthName=device+'-HOLEDEPTH'
    bitdepthName=device+'-BIT_DEPTH'

#wobName='01_Global-Generic_Surface-SWOB'
#rigstateName='01_Global-Generic_Surface-copRigState'


deeptime=(data.index).to_series()

#--------------------DH data---------------------------------------
#load EMS 1hz data
##bitter owen colin A
#emsdatafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/EMS/BITTERLY_OWENS_COLIN_A_ULW#1_BHA02R01_EMS01_SNEMSIB38_MEMORYMERGED.csv'
#emshifidatafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/EMS/bha02_merge_hifi.csv'
#timeshiftHour=0
#timeshiftSecond=204 


#
#RUCKMAN_RANCH_30
emsdatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_30/ems/merge_ems_1hz.csv'
emshifidatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_30/ems/merge_ems_hifi.csv'
timeshiftHour=5
timeshiftSecond=56

#RUCKMAN_RANCH_31
#emsdatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_31/ems_data/emsdata_allBHA_ruckman31.csv'
#emshifidatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_31/ems_data/ems50Hz_may1011.csv'
#
#timeshiftHour=5
#timeshiftSecond=50

#RUCKMAN_RANCH_32
#emsdatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_32/ems_data/merge_ems_1hz.csv'
#emshifidatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_32/ems_data/merge_ems_hifi.csv'
#timeshiftHour=5
#timeshiftSecond=35

##RUCKMAN_RANCH_34
#emsdatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_34/ems/RUCKMAN_RANCH_34_MEMORYMERGED.csv'
#emshifidatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANC NH_34/ems/merge_ems_hifi.csv'
#timeshiftHour=5
#timeshiftSecond=74

#RUCKMAN_RANCH_35
#emsdatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_35/ems/merge_ems_1hz.csv'
#emshifidatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_35/ems/merge_ems_hifi.csv'
#timeshiftHour=5
#timeshiftSecond=65


emsdata=deep_util.loadDEEPData(emsdatafile,timeColumnName='TIME(datetime)',
                               timeshiftHour=timeshiftHour, timeshiftSecond=timeshiftSecond)

emsdatahifi=deep_util.loadDEEPData(emshifidatafile,timeColumnName='TIME(datetime)',
                               timeshiftHour=timeshiftHour, timeshiftSecond=timeshiftSecond)


emsLatAccelXRaw=emsdata['EMS_LATX_MAX(G)']
emsLatAccelX = emsLatAccelXRaw[~np.isnan(emsLatAccelXRaw)]

emsLatAccelYRaw=emsdata['EMS_LATY_MAX(G)']
emsLatAccelY = emsLatAccelYRaw[~np.isnan(emsLatAccelYRaw)]

emsLatAccelTRaw=emsdata['EMS_LLAT_MAX(G)']
emsLatAccelT = emsLatAccelTRaw[~np.isnan(emsLatAccelTRaw)].resample('10ms').mean()
emsLatAccelT_aligned=emsLatAccelT.reindex(deeptime).interpolate()


emsRPM=emsdatahifi['RPM_GYRO(RPM)'].resample('10ms').mean()
emsRPM_aligned=emsRPM.reindex(deeptime).interpolate()

#ems_gx= emsdatahifi['GX(G)']
#ems_gy= emsdatahifi['GY(G)']
#ems_gz= emsdatahifi['AXIAL_VIBRATION(G)']
#
#ems_gx_100hz=ems_gx.resample('10ms').mean().interpolate(method='linear')
#ems_gx_aligned=ems_gx_100hz.reindex(deeptime).interpolate()
#
#ems_gy_100hz=ems_gy.resample('10ms').mean().interpolate(method='linear')
#ems_gy_aligned=ems_gy_100hz.reindex(deeptime).interpolate()
#
#ems_gz_100hz=ems_gz.resample('10ms').mean().interpolate(method='linear')
#ems_gz_aligned=ems_gz_100hz.reindex(deeptime).interpolate()

#%%
#high frequency

#isubDatafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/Isub/Bitterly_Owens_Colin-A_10252017.csv'
isubDatafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/Isub/isubdata_merge.csv'

isubDatahifi=deep_util.loadDEEPData(isubDatafile,timeColumnName='DateTime',
                               timeshiftHour=0, timeshiftSecond=0)

#ReadFromCsv=False
#if ReadFromCsv:
#
#    isubDatahifiraw=pd.read_csv(isubDatafile,skiprows=[1])
#    
#    isubDatahifiraw
#    
#    
#    timestamp_isub = pd.to_datetime(isubDatahifiraw['DateTime'],unit='s')
#    #timestamp_ems_shifted=timestamp_ems-pd.offsets.Second(78)-pd.offsets.Hour(5)
#    timestamp_isub_shifted=timestamp_isub-pd.offsets.Second(208)-pd.offsets.Hour(0)
#    
#    isubDatahifiraw['TIME(datetime)']=timestamp_isub_shifted
#    isubDatahifi=isubDatahifiraw.set_index('TIME(datetime)')
#    #isubDatahifi_aligned=isubDatahifi.reindex(time).interpolate()
#    
#    #pickleFileName=datafile+'.p'
#    #pickleFileName='/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/1hz/BITTERLY_OWENS_COLIN_A_ULW_1_merge.p'
#    with open(pickleFileName,'wb') as file:
#        pickle.dump(isubDatahifi,file)
#else:
#    t0 = time()
#    with open(pickleFileName,'rb') as file:
#        isubDatahifi=pickle.load(file)
#    print('done reading pickle in %.2fs.' % (time() - t0))

#bitdepthName='BITDEPTH(FT)' 

#emsWobName=['WOB(KLB)']
#rigstateName='01_Global-Generic_Surface-copRigState'


channelList=isubDatahifi.columns.tolist();


#
wobIsub=isubDatahifi[' WOB']
torqueIsub=isubDatahifi[' TOB']
drpmIsub=isubDatahifi[' RPMmag']
IsubDrpm_aligned=drpmIsub.reindex(deeptime).interpolate()
#%% set depth range and channels
depthMin=5000/3.28;
depthMax=18000/3.28;
mask= (data[holedepthName]>depthMin) & (data[holedepthName]< depthMax)
#mask= (data['01_Global.Pason_Surface.holeDepth']>depthMin) & (data['01_Global.Pason_Surface.holeDepth']< depthMax)
datacut = data.loc[mask]
#assign channels
torque=datacut['01_GLOBAL_PASON_TTS-STRQ']
tension=datacut['01_GLOBAL_PASON_TTS-TENSION']
drpm=datacut['01_GLOBAL_PASON_TTSV-DRPM']
holedepth = datacut[device+'-HOLEDEPTH']
rigstate=datacut[device+'-copRigState']
bitdepth=datacut[device+'-BIT_DEPTH']
srpm=datacut[device+'-SRPM']
wob=datacut[device+'-SWOB']
ar_tts=datacut['01_GLOBAL_PASON_TTS-RAD_ACCELERATION']
at_tts=datacut['01_GLOBAL_PASON_TTS-TAN_ACCELERATION']
az_tts=datacut['01_GLOBAL_PASON_TTS-AXIAL_ACCELERATION']

# use only for isub data
#drpm_data=IsubDrpm_aligned.loc[mask] 
drpm_data=emsRPM_aligned.loc[mask] 

aLat_ems=emsLatAccelT_aligned.loc[mask]
#%% quick look  
def quickPlot(data,channelNames):
    nchannel = len(data)
    plt.figure()
    for i in range(nchannel):
        plt.plot(data[i],label=channelNames[i])
    plt.legend()
    plt.show()
    
channelList=[drpm_data,torque/1350,aLat_ems*9.8]
channelNames=['bit rpm','torque','EMS lat accel']
quickPlot(channelList,channelNames)

#    
#%% test helper functions
#target=generateTarget(aLat_ems,ntSegment=4096,ntOverlap=2048,type='max')
#spec_torque=spectral.calc_spectrogram(torque,fs=100,ntSegment=4096,ntOverlap=2048)

#X,Y=generateSamples(torque,aLat_ems,fs=100,ntSample=30000,ntStep=30000,ntSegment=30000,ntOverlap=0)
#wobArray = generateTarget(swbb,ntSegment=30000,ntOverlap=20000,type='median')
#rigstateArray=generateTarget(rigstate,ntSegment=30000,ntOverlap=20000,type='median')
#X,Y=generateSamplesMultiChannel([torque,tension],aLat_ems,fs=100,ntSample=12000,ntStep=6000,ntSegment=4096,ntOverlap=2048)
#%% Gathering training data for vibration 
#import pdb
fs=100
ntSample=int(60*fs)
ntSampleStep=ntSample*0.5
ntSegment=1024
ntSegOverlap=ntSegment*0.75

inputChannel=['01_GLOBAL_PASON_TTS-STRQ', 
              '01_GLOBAL_PASON_TTS-TENSION', 
              '01_GLOBAL_GENERIC_SURFACE-SRPM',
              '01_GLOBAL_PASON_TTSV-DRPM']


#inputChannel=['torque', 'tension', 'srpm','drpm']

#inputData=[torque, tension, srpm, drpm]
inputData=[datacut[ch] for ch in inputChannel]

targetChannel=['aLat_ems']
targetData=aLat_ems
#Main data and target
#pdb.set_trace()
#X,Y=deep_util.generateSpecSamples(torque,aLat_ems,fs=fs,ntSample=ntSample,ntStep=ntSampleStep,ntSegment=ntSegment,ntOverlap=ntSegOverlap)
#X,Y=deep_util.generateSpecSamples(torque,rigstate,fs=fs,ntSample=ntSample,ntStep=ntSampleStep,ntSegment=ntSegment,ntOverlap=ntSegOverlap)
#X,Y=deep_util.generateSpecSampleMultiChannel([torque,tension,srpm],rigstate,fs=fs,ntSample=ntSample,ntStep=ntSampleStep,ntSegment=ntSegment,ntOverlap=ntSegOverlap)


Xt,Yt,Xs,Ys = deep_util.generateMixedSampleMultiChannel(inputData, targetData, fs, ntSample,
                                                        ntSampleStep, ntSegment, ntSegOverlap,
                                                        ifNormalize=True)

#auxillary data, one per sample
wobArray = deep_util.generateTarget(wob,ntSegment=ntSample,ntOverlap=ntSample-ntSampleStep,type='median')
rigstateArray=deep_util.generateTarget(rigstate,ntSegment=ntSample,ntOverlap=ntSample-ntSampleStep,type='median')
maxAccelArray=deep_util.generateTarget(aLat_ems,ntSegment=ntSample,ntOverlap=ntSample-ntSampleStep,type='max')
depthArray=deep_util.generateTarget(bitdepth,ntSegment=ntSample,ntOverlap=ntSample-ntSampleStep,type='median')

vf=np.linspace(0,fs/2,ntSegment//2+1)

#X,Y = deep_util.generateTimeSamplesMultiChannel([torque,tension,srpm],fs=fs,targetSeries=aLat_ems,ntSample=ntSample,ntStep=ntSample)
#%%  save  time data
dataDict={'ntSample':ntSample,'ntSampleStep':ntSampleStep,'ntSegment':ntSegment,'ntSegOverlap':ntSegOverlap,'wobArray':wobArray,
          'rigstateArray':rigstateArray,'maxAccelArray':maxAccelArray,'inputChannel':inputChannel,'targetChannel':targetChannel,
          'depthArray':depthArray,'Yt':Yt,'Xt':Xt}

#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/timeseries_60s_ColinA_bitRPM.p'
dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/timeseries_60s_ColinA_latAccel.p'

#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/timeseries_60s_RR30_latAccel.p'
with open(dataFile,'wb') as file:
    pickle.dump(dataDict,file,protocol=4)
#%% 
#make spec from time data
#Xt,Yt = deep_util.generateTimeSamplesMultiChannel([torque,tension,srpm],fs=fs,targetSeries=aLat_ems,ntSample=ntSample,ntStep=ntSampleStep)
Xs,Ys,vf = deep_util.makeSpecfromTSMultiChannel(Xt,Yt,fs, 512,128)

maxYs=[np.max(y) for y in Ys ]
maxYt=[np.max(y) for y in Yt ]
plt.figure()
plt.plot(depthArray,maxYs,'o',label='from Ys')
plt.plot(depthArray,maxYt,'o')
plt.plot(depthArray,np.array(maxYt)-np.array(maxYs),'o')

plt.legend()

#%%  save mixed spec and time data
dataDict={'ntSample':ntSample,'ntSampleStep':ntSampleStep,'ntSegment':ntSegment,'ntSegOverlap':ntSegOverlap,'wobArray':wobArray,
          'rigstateArray':rigstateArray,'maxAccelArray':maxAccelArray,'inputChannel':inputChannel,'targetChannel':targetChannel,
          'depthArray':depthArray,'Yt':Yt,'Xt':Xt,'Ys':Ys,'Xs':Xs,'vf':vf}
dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/mixed_RR30_aLat.p'
with open(dataFile,'wb') as file:
    pickle.dump(dataDict,file,protocol=4)
#%%  save spec data
    
#print('done writing pickle in %.2fs.' % (time() - t0))
dataDict={'ntSample':ntSample,'ntSampleStep':ntSampleStep,'ntSegment':ntSegment,'ntSegOverlap':ntSegOverlap,'wobArray':wobArray,
          'rigstateArray':rigstateArray,'maxAccelArray':maxAccelArray,'inputChannel':inputChannel,'targetChannel':targetChannel,
          'depthArray':depthArray,'Y':Y,'X':X}
#dataOnlyFile='/home/zhay/DEEP/machineLearning/stickslipDetector/trainingData_11400-11650ft_xy_new.p'
#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/spec_torque_accel_ColinA.p'
#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/spec_torque_accel_RR30.p'
#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/spec_torque_accel_RR30_60sec.p'
dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/spec_torque_accel_RR30_rigstate.p'

#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/spec_torque_accel_RR31.p'
#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/spec_torque_accel_RR32.p'
#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/spec_torque_accel_RR34.p'
#dataFile='/home/zhay/DEEP/machineLearning/vibrationDetector/spec_torque_accel_RR35.p'

with open(dataFile,'wb') as file:
    pickle.dump(dataDict,file,protocol=4)
 #print('done writing pickle in %.2fs.' % (time() - t0))
#
#
# #%%
# idx=np.random.randint(0,len(X))
# deep_util.plotSpecExample(X[idx],Y[idx],fs=fs,ntSample=ntSample)
