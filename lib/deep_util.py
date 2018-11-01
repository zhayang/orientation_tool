#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:43:08 2018

@author: zhay
"""

#DEEP util function

import scipy as sp
#import pybrain as pb
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy import signal
from time import time
import os
import pickle
import spectral
import re
#%%  data IO utility functions
def loadDEEPData(datafile,timeColumnName='DATETIME',timeshiftHour=5,timeshiftSecond=0,savePickle=False):
    pickleFileName=datafile+'.p'
    if os.path.isfile(pickleFileName)==False:
        print('pickle file does not exist, load from csv')
    
        dataraw=pd.read_csv(datafile,skiprows=[1])
    #    dataraw        
    #% filter data and change index to timestamp
        timestamp = pd.to_datetime(dataraw[timeColumnName],errors='coerce')
        timestamp=timestamp-pd.offsets.Second(timeshiftSecond)-pd.offsets.Hour(timeshiftHour)
        dataraw[timeColumnName]=timestamp                        
        data=dataraw.set_index(timeColumnName)
        del dataraw
        pickleFileName=datafile+'.p'
        t0 = time()
        if savePickle:
            with open(pickleFileName,'wb') as file:
                pickle.dump(data,file,protocol=4)
            print('done writing pickle in %.2fs.' % (time() - t0))
    else:
        print('load from pickle file')
        t0 = time()
        with open(pickleFileName,'rb') as file:
            data=pickle.load(file)
        print('done reading pickle in %.2fs.' % (time() - t0))
    return data
#%% Utility functions
# function to insert ones to the end of positive labels
def insert_ones(z,spread):
#    nsample=z.shape[0]
    ind_end=np.where(np.diff(z)<0)[0]
    for ind in ind_end:
        z[ind+1:ind+spread+1]=1
    return z

# function to generate binary labels based on threshold
def distributeTarget(Y,highlimit=1,spread=5):
    """
    input: Y nsample x nt input data 
        
    """
    nsample=Y.shape[0] 
#    nt=Y.shape[1]
    Z=np.zeros((Y.shape[0],Y.shape[1]))

    for j in range(nsample):
        Z[j,Y[j,:]>=highlimit]=1
        Z[j,:] = insert_ones(Z[j,:],spread)
    return Z

# function to generate binary target from y if value exceed a certain threshold
def calcThresholdTarget(y,threshold=0,type='absolute'):
    if type =='absolute':
        return np.where(np.abs(y)>threshold,1,0)
    else:
        return np.where(y>threshold,1,0)

def calcThresholdTargetHL(y,thresholdLow=0, thresholdHigh=1,type='absolute'):
    if type =='absolute':
        return np.where(np.abs(y)>thresholdHigh,1,np.where(np.abs(y)<thresholdLow,-1,0))
    else:
        return np.where(y>thresholdHigh,1,np.where(y<thresholdLow,-1,0))
    

def divide_train_test(X, Y, trainPerc=0.8, shuffle=True):
    """Prepare training and test data with a ratio"""
    nsample = X.shape[0]
    idxArray = np.arange(0, nsample)
    if shuffle:
        np.random.seed(8)
        np.random.shuffle(idxArray)
    idx_train = idxArray[0:np.int(nsample * trainPerc)]
    idx_test = idxArray[np.int(nsample * trainPerc):]

    X_train = X[idx_train]
    Y_train = Y[idx_train]
    X_test = X[idx_test]
    Y_test = Y[idx_test]

    return X_train, Y_train, X_test, Y_test
#%% functions to generate data from continuous time series

def generateTargetBinary(targetTimeSeries,ntSegment,ntOverlap=0,threshold=0,type='absolute'):
    """ function to generate binary labels from timeseries with threshold"""
    z=calcThresholdTarget(targetTimeSeries,threshold,type)
    return generateTarget(z,ntSegment,ntOverlap)
    
def generateTarget(targetTimeSeries,ntSegment,ntOverlap,type='max'):
    """function to output key statistics during fixed time intervals from time series"""
    ntTot=len(targetTimeSeries)
    itStart=0
    itEnd=ntSegment
    nwin=0
    result=[]
    while itEnd<=ntTot:
#        print('------')
#        print(itStart)
#        print(itEnd)
        y_win=np.array(targetTimeSeries[itStart:itEnd])
        if type=='max':
            stat=np.nanmax(y_win)
        elif type=='min':
            stat=np.nanmin(y_win)
        elif type=='mean':
            stat=np.nanmean(y_win)
        elif type=='std':
            stat=np.nanstd(y_win)
        elif type=='median':
            stat=np.nanmedian(y_win)
#        add stat to results
        result.append(stat)
        
#        move to next window
        itStart=int(itStart+ntSegment-ntOverlap)
        itEnd=int(itStart+ntSegment)
        nwin+=1
#   return list of result
    output=np.array(result)
    return output


def generateOneSpecSample(inputTimeSeries, fs, ntSegment, ntOverlap):
    """calculate spectrogram from time series"""
    vf,spec=spectral.calc_spectrogram(np.array(inputTimeSeries),fs=fs,ntSegment=ntSegment,ntOverlap=ntOverlap)
    return vf,spec

#def
def generateSpecSamples(inputSeries, targetSeries, fs, ntSample, ntStep, ntSegment, ntOverlap):
    
    """generate lists of data samples"""
    ntTotal=targetSeries.shape[0]
#    nwin=(ntTotal-ntwin)//ntStep
    X=[]
    Y=[]
    itBegin=0
    itEnd=ntSample
    iwin=0
#    for iwin in range(0,nwin):
    
#        itBegin=iwin*ntStep
#        itEnd = itBegin + ntwin
    while itEnd<=ntTotal:
        inputData=np.array(inputSeries[itBegin:itEnd])
        targetData=np.array(targetSeries[itBegin:itEnd])
#        spectrogram sample
        _,spec=generateOneSpecSample(inputData, fs, ntSegment, ntOverlap)
#        target data
        target=generateTarget(targetData,ntSegment,ntOverlap,type='max')
        X.append(spec)
        Y.append(target)
        itBegin=int(itBegin+ntStep)
        itEnd=(itBegin+ntSample)
#        print(itBegin)
        iwin+=1
        
        if(iwin%100==0):
            print('processing window %3.0f'%iwin)
        
    return X,Y


def generateMixedSampleMultiChannel(inputSeriesList, targetSeries, fs, ntSample, ntStep, ntSegment, ntOverlap, ifNormalize=False):
    """Generate mixed sampled from multiple timeseries"""
    Xtime,Ytime = generateTimeSamplesMultiChannel(inputSeriesList,targetSeries,fs,ntSample,ntStep,ifNormalize)
    Xspec,Yspec = generateSpecSamplesMultiChannel(inputSeriesList,targetSeries,fs,ntSample,ntStep,ntSegment,ntOverlap)    
    return Xtime,Ytime,Xspec,Yspec


def generateSpecSamplesMultiChannel(inputSeriesList,targetSeries,fs,ntSample,ntStep,ntSegment,ntOverlap):
    """generate lists of spec data samples"""
    nChannel=len(inputSeriesList)
    ntTotal=targetSeries.shape[0]
    iwin=0
    X=[]
    Y=[]
    itBegin=0
    itEnd=ntSample
    while itEnd<=ntTotal:
#        for iwin in range(0,nwin):
#        itBegin=iwin*ntStep
#        itEnd = itBegin + ntSample
        
#       target
        targetData=np.array(targetSeries[itBegin:itEnd])
        target=generateTarget(targetData,ntSegment,ntOverlap,type='max')
        
#        shape of the target (nsegment, 1)
#        targetShape=target.shape
        specList=[]
        for ichannel in range(0,nChannel):
            
            inputData=np.array(inputSeriesList[ichannel][itBegin:itEnd])
            vf,spec=generateOneSpecSample(inputData, fs, ntSegment, ntOverlap)
            specList.append(spec)
        specArray=np.zeros((spec.shape[0],spec.shape[1],nChannel))
        
#        convert list to array after knowing the channels
        for ichannel in range(0,nChannel):
            specArray[:,:,ichannel]=specList[ichannel]
            
        itBegin=int(itBegin+ntStep)
        itEnd=int(itBegin+ntSample)
        iwin+=1
        
        X.append(specArray)
        Y.append(target)
        if(iwin%100==0):
            print('processing window %3.0f'%iwin)
    return X,Y

def makeSpecfromTSMultiChannel(Xt,Yt,fs,ntSegment,ntOverlap,ytype='max'):
    """generate spectra data from multi-channel time series data, for each time series, calculate spectra from overlapping
    windows of size ntSegment and overlap by ntOverlap
    input: Xt,Yt: list of nsample timeseries samples of size (, nt, nchannel) and (, nt)
    output Xs, Ys: list of nsample spec samples of size (nstep, nf,nchannel) and (,nstep)"""
    assert(len(Xt)==len(Yt))
    nsample = len (Yt)
    nt,nchannel = Xt[0].shape
    Xs=[]
    Ys=[]
    nextLevel=0
    for i in range(nsample):
        # if i*10//nsample==0:
            # print('Processed %2.2f percent of data'%(i*100//nsample))
        # make spectrogram
        speclist=[]
        for ic  in range(nchannel):
            inputData=np.array(Xt[i][:,ic])
            vf,spec=generateOneSpecSample(inputData, fs, ntSegment, ntOverlap)
            speclist.append(spec)
        specArray=np.zeros((spec.shape[0],spec.shape[1],nchannel))
        for ic in range(0,nchannel):
            specArray[:,:,ic]=speclist[ic]
        # make target
        target=generateTarget(Yt[i],ntSegment,ntOverlap,type=ytype)
        Xs.append(specArray)
        Ys.append(target)
        perc = i / nsample * 100
        if perc > nextLevel:
            print('%2.0f percent finished' % perc)
            nextLevel+=10
    return Xs,Ys,vf


def generateTimeSamplesMultiChannel(inputSeriesList,targetSeries,fs,ntSample,ntStep,ifNormalize=False):
    
    """generate lists of time series samples"""
    nChannel=len(inputSeriesList)
    ntTotal=targetSeries.shape[0]
    iwin=0
    X=[]
    Y=[]
    itBegin=0
    itEnd=ntSample
    while itEnd<=ntTotal:

        targetData=np.array(targetSeries[itBegin:itEnd])
        inputData=np.zeros((ntSample,nChannel))
        for ichannel in range(0,nChannel):            
            x= np.array(inputSeriesList[ichannel][itBegin:itEnd])  
            if ifNormalize:
                inputData[:,ichannel]=normalizeData(x,axis=0)
            else:
                inputData[:,ichannel]=x        
#        convert list to array after knowing the channels
                

        itBegin=int(itBegin+ntStep)
        itEnd=int(itBegin+ntSample)
        iwin+=1
        X.append(inputData)
        Y.append(targetData)
        if(iwin%100==0):
            print('processing window %3.0f'%iwin)
    return X,Y

def normalizeData(dataIn,removeMean=True,normalizeAmp=True,epsilon=0.0001,axis=1):
    dataOut=dataIn
    if(removeMean):
        dataOut=dataOut-np.nanmean(dataIn,axis=axis,keepdims=True)
    if(normalizeAmp):
        dataOut=dataOut/(np.nanstd(dataIn,axis=axis,keepdims=True)+epsilon)
    #    normalize data for training
    return dataOut


def generateSamples_backup(inputSeries,targetSeries,fs,ntSample,ntStep,ntSegment,ntOverlap):
    
    """generate lists of data samples"""
    ntTotal=targetSeries.shape[0]
    nwin=(ntTotal-ntSample)//ntStep
    X=[]
    Y=[]
    for iwin in range(0,nwin):
        itBegin=iwin*ntStep
        itEnd = itBegin + ntSample
        
        inputData=np.array(inputSeries[itBegin:itEnd])
        targetData=np.array(targetSeries[itBegin:itEnd])
#        print(len(inputData))

        _,spec=generateOneSpecSample(inputData, fs, ntSegment, ntOverlap)
        target=generateTarget(targetData,ntSegment,ntOverlap,type='max')
        X.append(spec)
        Y.append(target)
#        print(itBegin)

        if(iwin%100==0):
            print('processing window %3.0f'%iwin)
        
    return X,Y


def generateSamplesMultiChannel_backup(inputSeriesList,targetSeries,fs,ntSample,ntStep,ntSegment,ntOverlap):
    
    """generate lists of data samples"""
    nChannel=len(inputSeriesList)
    ntTotal=targetSeries.shape[0]
    nwin=(ntTotal-ntSample)//ntStep
    X=[]
    Y=[]
    for iwin in range(0,nwin):
        itBegin=iwin*ntStep
        itEnd = itBegin + ntSample
        
#        target
        targetData=np.array(targetSeries[itBegin:itEnd])
        target=generateTarget(targetData,ntSegment,ntOverlap,type='max')
        
#        shape of the target (nsegment, 1)
#        targetShape=target.shape
        specList=[]
        for ichannel in range(0,nChannel):
            
            inputData=np.array(inputSeriesList[ichannel][itBegin:itEnd])
            vf,spec=generateOneSpecSample(inputData, fs, ntSegment, ntOverlap)
            specList.append(spec)
        specArray=np.zeros((spec.shape[0],spec.shape[1],nChannel))
        
#        convert list to array after knowing the channels
        for ichannel in range(0,nChannel):
            specArray[:,:,ichannel]=specList[ichannel]

        X.append(specArray)
        Y.append(target)
        if(iwin%100==0):
            print('processing window %3.0f'%iwin)
    return X,Y

def plotSpecExample(x,y,fs,ntSample):
    dt=1/fs
    nf,nt=x.shape
    vf=np.linspace(0,fs/2,nf)
    vt=np.linspace(0,ntSample*dt,nt)
    vec_t=np.arange(0,len(y))*dt

    
    plt.figure()
    ax1=plt.subplot(211)
    extents = vt[0], vt[-1],vf[0],vf[-1]
    
    plt.imshow(10*np.log10(x),aspect='auto',origin='lower',extent=extents)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
#    plt.clim(80,120)
    plt.ylim(0,10)
#    plt.show()
    
    ax2=plt.subplot(212,sharex=ax1)
    plt.plot(vt,y,label='target')
    ax2.legend()
#    plt.show()
#    ax2b=ax2.twinx()
#    plt.legend()
  
    #%% calculate dogleg 
def calcDogleg(svymd,svyinc,svyazm,nsmooth=0):
    deltaAzm=np.append(0,np.diff(svyazm))
    deltaInc =np.append(0,np.diff(svyinc))
    ind_svy = ((deltaAzm!=0) & ~np.isnan(deltaAzm)& (deltaInc!=0) & ~np.isnan(deltaInc))
    
#    select valid and unique survey points
    deltaIncArray=deltaInc[ind_svy]                                                                                                                                            
    deltaAzmArray=deltaAzm[ind_svy]
    incArray=svyinc[ind_svy]
    mdArray=svymd[ind_svy]
    deltaMdArray=np.append(np.diff(mdArray)[0],np.diff(mdArray))
    deltaAngle=np.arccos(np.sin(incArray)*np.sin(incArray-deltaIncArray)*np.cos(deltaAzmArray)+ np.cos(incArray)*np.cos(incArray-deltaIncArray))
    dls=deltaAngle/deltaMdArray*(180/np.pi)*(100/3.28)
    dls_smooth=dls.rolling(nsmooth).mean()
    return dls_smooth, mdArray
 
#   calculate dogleg with NaN in surveys
def calcDoglegWithNan(svymd,svyinc,svyazm,nsmooth=0):
#    remove nan from data
    ind_valid= (~np.isnan(svyinc) & ~np.isnan(svyazm) & ~np.isnan(svymd) & (np.abs(svyinc)<2*np.pi) & (np.abs(svyazm)<2*np.pi))
    svyinc=svyinc[ind_valid]
    svyazm=svyazm[ind_valid]
    svymd=svymd[ind_valid]
    
#    remove 0 from data, only choose when survey are updated
    deltaAzm=np.append(0,np.diff(svyazm))
    deltaInc =np.append(0,np.diff(svyinc))
    ind_svy = ((deltaAzm!=0) & ~np.isnan(deltaAzm)& (deltaInc!=0) & ~np.isnan(deltaInc))
#    select valid and unique survey points
    deltaIncArray=deltaInc[ind_svy]
    deltaAzmArray=deltaAzm[ind_svy]
    incArray=svyinc[ind_svy]
    
    mdArray=svymd[ind_svy]
    deltaMdArray=np.append(np.diff(mdArray)[0],np.diff(mdArray))
    deltaAngle=np.arccos(np.sin(incArray)*np.sin(incArray-deltaIncArray)*np.cos(deltaAzmArray)+ np.cos(incArray)*np.cos(incArray-deltaIncArray))
    dls=deltaAngle/deltaMdArray*(180/np.pi)*(100/3.28)
    dls_smooth=dls.rolling(nsmooth).mean()
    
    return dls_smooth, mdArray

#   calculate dogleg and align survey data
def alignSurvey(svymd,svyinc,svyazm,mu,nsmooth=0):
#    remove nan from data
    ind_valid= (~np.isnan(svyinc) & ~np.isnan(svyazm) & ~np.isnan(svymd) & (np.abs(svyinc)<2*np.pi) & (np.abs(svyazm)<2*np.pi))
    svyinc=svyinc[ind_valid]
    svyazm=svyazm[ind_valid]
    svymd=svymd[ind_valid]
    muTmp=mu[ind_valid]
    
#    remove 0 from data, only choose when survey are updated
    deltaAzm=np.append(0,np.diff(svyazm))
    deltaInc =np.append(0,np.diff(svyinc))
    ind_svy = ((deltaAzm!=0) & ~np.isnan(deltaAzm)& (deltaInc!=0) & ~np.isnan(deltaInc))
#    select valid and unique survey points
    deltaIncArray=deltaInc[ind_svy]
    deltaAzmArray=deltaAzm[ind_svy]
    
    incArray=svyinc[ind_svy]
    azmArray=svyazm[ind_svy]
    mdArray=svymd[ind_svy]
    muArray=muTmp[ind_svy]
    
    deltaMdArray=np.append(np.diff(mdArray)[0],np.diff(mdArray))
    deltaAngle=np.arccos(np.sin(incArray)*np.sin(incArray-deltaIncArray)*np.cos(deltaAzmArray)+ np.cos(incArray)*np.cos(incArray-deltaIncArray))
    dls=deltaAngle/deltaMdArray*(180/np.pi)*(100/3.28)
    
    dls_smooth=dls.rolling(nsmooth).mean()
    
    return dls_smooth, mdArray,incArray,azmArray,muArray
class FeatureScaler4D:
    '''A feature scaler for 4D input data'''
    def __init__(self,channelDim):
        self.meanArray=None
        self.stdArray=None
        self.sampleAxis=0
        self.channelDim=channelDim
        self.nchannel=None

    def fit(self,X):
        # get number of channels
        assert self.channelDim<len(X.shape)

        self.nchannel=X.shape[self.channelDim]
        self.meanArray=np.zeros((self.nchannel,))
        self.stdArray=np.zeros((self.nchannel,))
        # get std and mean for different data dimensions
        for ic in range(self.nchannel):
            if self.channelDim==1:
                self.meanArray[ic]=np.mean(X[:,ic,:,:].ravel())
                self.stdArray[ic]=np.std(X[:,ic,:,:].ravel())
            elif self.channelDim==2:
                self.meanArray[ic]=np.mean(X[:,:,ic,:].ravel())
                self.stdArray[ic]=np.std(X[:,:,ic,:].ravel())
            elif self.channelDim==3:
                self.meanArray[ic] = np.mean(X[:, :, :,ic].ravel())
                self.stdArray[ic] = np.std(X[:, :, :,ic].ravel())

    def transform(self,X):
        Xnorm = np.zeros(X.shape)
        nchannel=X.shape[self.channelDim]
        assert nchannel==self.nchannel

        for ic in range(self.nchannel):
            if self.channelDim==1:
                Xnorm[:,ic,:,:]=(X[:,ic,:,:]-self.meanArray[ic])/self.stdArray[ic]
            elif self.channelDim==2:
                Xnorm[:,:,ic,:]=(X[:,:,ic,:]-self.meanArray[ic])/self.stdArray[ic]
            elif self.channelDim==3:
                Xnorm[:,:,:,ic]=(X[:,:,:,ic]-self.meanArray[ic])/self.stdArray[ic]

        return Xnorm

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self,Xnorm):
        X = np.zeros(Xnorm.shape)
        nchannel=Xnorm.shape[self.channelDim]
        assert nchannel==self.nchannel

        for ic in range(self.nchannel):
            if self.channelDim==1:
                X[:,ic,:,:]=Xnorm[:,ic,:,:]*self.stdArray[ic]+self.meanArray[ic]
            elif self.channelDim==2:
                X[:,:,ic,:]=Xnorm[:,:,ic,:]*self.stdArray[ic]+self.meanArray[ic]
            elif self.channelDim==3:
                X[:,:,:,ic]=Xnorm[:,:,:,ic]*self.stdArray[ic]+self.meanArray[ic]
        return X

class FeatureGenerator:
    """ a class to process time series and genrate labeled featrue and target samples for trainng"""
    def __init__(self,dt=0.01):
        self.dt=dt
        self.fs=1/dt
        self.freqMax=None
        self.nfft=None
        self.nfftOverlap=0
        # self.selectCondition=None
        self.batchSize=None

    def setParam(self,nfft,nfftOverlap,freqMax,batchSize,conditionList=[]):
        self.nfft=nfft
        self.nfftOverlap=nfftOverlap
        self.freqMax=freqMax
        # self.selectCondition=conditionList
        self.batchSize=batchSize
    # todo function to read data from files
    def GenerateFeatures(self,fileList,conditionList,ytype='mean',cutToBatch=True):
        X_time, Y_time, auxDataDict = self.loadRawData(fileList)
        Xt_select, Yt_select, auxSelect = self.selectData(X_time, Y_time, auxDataDict, conditionList)
        # print('selected %5.2f samples'Xt_select.shape[0])
        Xs_select, Ys_select = self.calcSpec(Xt_select, Yt_select, ytype)
        X, Y, auxDataDict = self.postProcess(Xt_select, Yt_select, Xs_select, Ys_select, auxSelect,cutToBatch)
        return X,Y,auxDataDict

    # ----------------Private methods-------------------------------------------------

    def loadRawData(self,fileList):
        Xlist = []
        Ylist = []
        auxDataDict = {}
        for datafile in fileList:
            with open(datafile, 'rb') as file:
                dataDict = pickle.load(file)

            X_time = dataDict['Xt']
            Y_time = dataDict['Yt']
            auxChannelList = list(dataDict['auxDataDict'].keys())


            #  merge
            Xlist = Xlist + X_time
            Ylist = Ylist + Y_time
            for ch in auxChannelList:
                if ch in auxDataDict.keys():
                    auxDataDict[ch] = np.concatenate((auxDataDict[ch], dataDict['auxDataDict'][ch]))
                else:
                    auxDataDict[ch] = dataDict['auxDataDict'][ch]
            print('Reading file ' + datafile)
            print('reading %4f samples' % len(Y_time))

        print('Size of merged data X is: ')
        Xt = np.array(Xlist)
        Yt = np.array(Ylist)
        print('number of samples')
        print(len(Xt))
        print('Shape of X (nsample,nt,nchannel):')
        print(Xt.shape)

        ntSample = dataDict['ntSample']
        ntSegment = dataDict['ntSegment']
        return Xt, Yt, auxDataDict
    #todo select subset of data
       # return X,Y


    def selectData(self,X_time,Y_time,auxDataDict,conditionList=[]):
        """ select subset of data
        auxDataDict: {channel: array of values}
        conditionList: list of conditions with in the order of 'channelName, upppr bound, lower bound' """
        xtmp = np.reshape(X_time, (X_time.shape[0], -1))
        isvalid_x = ~np.isnan(xtmp).any(axis=1)
        isvalid_y = ~np.isnan(Y_time).any(axis=1)
        # self.selectCondition=conditionList
        # valid data for x and y
        ind_select= (isvalid_x) & (isvalid_y)

        # go through each conditions
        for condition in conditionList:
            name,lowerlimit,upperlimit = condition[0],condition[1],condition[2]
            ind_select=ind_select & (auxDataDict[name]>=lowerlimit) & (auxDataDict[name]<=upperlimit)

        Xt_select = X_time[ind_select]
        Yt_select = Y_time[ind_select]
        auxData_select = {}
        for ch in auxDataDict.keys():
            auxData_select[ch] = auxDataDict[ch][ind_select]

        return Xt_select,Yt_select,auxData_select

    # def generateSpec
    def calcSpec(self,Xt_select,Yt_select,ytype='mean'):
        """calculate spectra from time series data and apply filter"""
        Xs_select, Ys_select, vf = makeSpecfromTSMultiChannel(Xt_select, Yt_select,
                                                              self.fs, ntSegment=self.nfft,
                                                              ntOverlap=self.nfftOverlap,
                                                              ytype=ytype)
        nf, ntSpec, nc = Xs_select[0].shape

        vf = np.linspace(0, self.fs / 2, nf)
        # vt=np.linspace(0,ntSample*dt,nt)
        df = vf[1] - vf[0]
        # only use frequency up to 5Hz
        ind_freq = vf <= self.freqMax
        Xs_select = np.array(Xs_select)
        Ys_select = np.array(Ys_select)

        Xs_select = Xs_select[:, ind_freq, :, :]

        return Xs_select, Ys_select

    def postProcess(self,Xt,Yt,Xs,Ys,auxData_select,cutToBatch=True):
        # reshape/merge/cut_to batch/normalize
        Ys = np.reshape(Ys, Ys.shape + (1,))
        Yt = np.reshape(Yt, Yt.shape + (1,))

        Xt = np.reshape(Xt, (Xt.shape + (1,)))
        Xs = np.transpose(Xs, (0, 2, 1, 3))

        if cutToBatch:
            Xs, Ys = self.cut_to_batch(Xs, Ys)
            Xt, Yt = self.cut_to_batch(Xt, Yt)
            for ch in auxData_select.keys():
                auxData_select[ch],_ = self.cut_to_batch(auxData_select[ch],None)

        X=[Xt,Xs]
        Y=[Yt,Ys]

        return X,Y,auxData_select
    # todo
    def normalize(self,X,Y,XScale=1,YScale=1):
        X/=XScale
        Y/=YScale
        return X,Y
    # todo
    def cut_to_batch(self, features, labels):
        # Needed for stateful training: make sure num samples is commensurate with batch size
        batchMultiple = np.int(np.float(features.shape[0]) / self.batchSize) * self.batchSize
        features = features[:batchMultiple]
        if labels is not None:
            labels = labels[:batchMultiple]
        return features, labels

    # todo
    #  separate train and test

class DataSelector:
    """A class to generate training and test data examples from a set of parameters """
    def __init__(self,dt=0.01):
        self.dt=dt
        self.fs=1/dt
        self.hasData=False
        self.inputChannelList=None
        self.targetChannelList=None
        self.dataFileName=None
        self.dhDataInfo=None
        # self.data=None
        self.datacut=None
        self.dhdata=None
        self.procSetting=None
        self.outFileName=None
        self.deeptime=None
        self.minDepth=0
        self.maxDepth=np.inf

    def setProcessingParam(self,ntSample,ntSampleStep,ntSegment,ntSegOverlap,depthMin,depthMax):
        self.procSetting={'ntSample':ntSample,'ntSampleStep':ntSampleStep,
                          'ntSegment':ntSegment,'ntSegOverlap':ntSegOverlap,
                          'depthMin':depthMin,'depthMax':depthMax}



    def setDHDataSource(self,name,timeShiftHour=0,timeShiftSecond=0,timeColumnName=None):
        self.dhDataInfo={'name':name,'timeShiftHour':timeShiftHour,
                         'timeShiftSecond':timeShiftSecond,'timeColumnName':timeColumnName}

    def loadData(self,datafile):
        # load deep data
        self.dataFileName=datafile
        self.inputChannelList=datafile
        data_full=loadDEEPData(datafile,savePickle=True)
        if ('01_Global-Generic_Surface-HOLEDEPTH' in data_full.columns):
            device = '01_Global-Generic_Surface'
            holedepthName = device + '-HOLEDEPTH'
            bitdepthName = device + '-BIT_DEPTH'
        else:
            device = '01_GLOBAL_GENERIC_SURFACE'
            holedepthName = device + '-HOLEDEPTH'
            bitdepthName = device + '-BIT_DEPTH'

        self.hasData=True
        mask = (data_full[holedepthName] > self.procSetting['depthMin']) \
               & (data_full[holedepthName] < self.procSetting['depthMax'])
        # mask= (data['01_Global.Pason_Surface.holeDepth']>depthMin) & (data['01_Global.Pason_Surface.holeDepth']< depthMax)
        self.datacut = data_full.loc[mask]
        self.deeptime = (self.datacut.index).to_series()


    def loadDHData(self):
        self.dhdata = loadDEEPData(self.dhDataInfo['name'], timeColumnName=self.dhDataInfo['timeColumnName'],
                                         timeshiftHour=self.dhDataInfo['timeShiftHour'],
                                   timeshiftSecond=self.dhDataInfo['timeShiftSecond'])


    # todo: allow target data from both input surface data and DH data
    def preprocess(self,inputChannelList=[],targetChannelList=[]):
        """pre-select valid data channels
        :param inputChannelList:
        :param targetChannelList:
        :return: inputDataList,inputChannelListUsed,targetDataList,targetChannelListUsed
        """
        inputDataList=[]
        inputChannelListUsed=[]
        targetDataList=[]
        targetChannelListUsed=[]
        for ch in inputChannelList:
            if ch in self.datacut.columns:
                inputDataList.append(self.datacut[ch])
                inputChannelListUsed.append(ch)
            else:
                print('skipping unknown input channel '+ ch)
        for ch in targetChannelList:
            if ch in self.dhdata.columns:
                y_raw=self.dhdata[ch]
                y_aligned=y_raw[~np.isnan(y_raw)].resample('10ms').mean().reindex(self.deeptime).interpolate()
                targetDataList.append(y_aligned)
                targetChannelListUsed.append(ch)
            else:
                print('skipping unknown target channel '+ ch)

        return  inputDataList,inputChannelListUsed,targetDataList,targetChannelListUsed

    def makedata(self,inputChannelList=[],targetChannelList=[],auxChannelList=[],calcSpec=False,normalize=False):

        # preprocess
        inputDataList, inputChannelListUsed, targetDataList, targetChannelListUsed = \
            self.preprocess(inputChannelList,targetChannelList)

        # make input and target data
        if calcSpec:
            Xt, Yt, Xs, Ys = generateMixedSampleMultiChannel(inputDataList, targetDataList[0], self.fs,
                                                             self.procSetting['ntSample'],
                                                             self.procSetting['ntSampleStep'],
                                                             self.procSetting['ntSegment'],
                                                             self.procSetting['ntSegOverlap'],
                                                              ifNormalize=normalize)
        else:
            Xt,Yt=generateTimeSamplesMultiChannel(inputDataList, targetDataList[0],self.fs,
                                                self.procSetting['ntSample'],
                                                self.procSetting['ntSampleStep'],
                                                ifNormalize=normalize)

        # make auxilary data
        auxDataDict=self.makeAuxData(auxChannelList)

        # make output dictionary
        if calcSpec:
            dataDict={'ntSample':self.procSetting['ntSample'],'ntSampleStep':self.procSetting['ntSampleStep'],
                      'ntSegment':self.procSetting['ntSegment'],'ntSegOverlap':self.procSetting['ntSegOverlap'],
                      'inputChannel':inputChannelListUsed,'targetChannel':targetChannelListUsed,
                      'auxDataDict':auxDataDict,
                      'Yt':Yt,'Xt':Xt,'Ys':Ys,'Xs':Xs}
        else:
            dataDict={'ntSample':self.procSetting['ntSample'],'ntSampleStep':self.procSetting['ntSampleStep'],
                      'ntSegment':self.procSetting['ntSegment'],'ntSegOverlap':self.procSetting['ntSegOverlap'],
                      'inputChannel':inputChannelListUsed,'targetChannel':targetChannelListUsed,
                      'auxDataDict':auxDataDict,
                      'Yt':Yt,'Xt':Xt}
        return dataDict

    def makeAuxData(self,auxChannelList):
        """Make auxiliary data : one point per sample
        :param auxChannelList: List of auxilary data channels
        :return: Distionary of axilary data and channels
        """
        # dataList=[]
        # channelList=[]
        dataDict={}
        for ch in auxChannelList:
            if ch in self.datacut.columns:
                y = generateTarget(self.datacut[ch],ntSegment=self.procSetting['ntSample'],
                                   ntOverlap=self.procSetting['ntSample']-self.procSetting['ntSampleStep'],
                                   type='median')
                dataDict[ch]=y
            else:
                print(' No channel named ' + ch)

        return dataDict

    def saveToFile(self,outFileName,dataDict):
        with open(outFileName, 'wb') as file:
            pickle.dump(dataDict, file, protocol=4)

    def resetData(self):
        self.hasData=False
        self.dataFileName=None
        self.dhDataInfo=None
        self.datacut=None
        self.dhdata=None
        self.deeptime=None
    # def preprocess(self):
    #     return None
    #
    # def makeDataSamples(self):
    #     return None

    # class ProcessingSetting(object):
    #     """class to hold processing parameters"""
    #     def __init__(self,):

    # class DHFile(object):
    #     """class structure to hold data file name and clock drift"""
    #     def __init__(self, name,timeShiftHour=0,timeShiftSecond=0,timeColumnName=None):
    #         self.name=name
    #         self.timeShiftHour=timeShiftHour
    #         self.timeShiftSecond=timeShiftSecond
    #         self.timeColumnName=timeColumnName


    def standardName(self,ch):
        """change to lower case and replace '-' with '_'  """
        if(type(ch)==str):
            return re.sub('-','_',ch).lower()
        else:
            return [re.sub('-','_',s).lower() for s in ch]
