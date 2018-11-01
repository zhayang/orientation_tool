from typing import Any, Union, List

import scipy as sp
import pandas as pd
import numpy as np

from numpy.core.multiarray import ndarray
from pandas import DataFrame

import deep_util
from time import time
import matplotlib.pyplot as plt
from scipy import signal
# from spectral import calc_coherence,calc_spec
# from scipy.signal import butter, lfilter, filtfilt
import pickle
import os

#from keras.models import load_model, save_model
# import h5py

class DrillingDataFrame:
    """A class to hold, manupulate and visualize drilling data time series """
    # df: dataframe containing drilling data

    def __init__(self,dt=1.0):
        self.dt=dt
        # self.setDefaultChannelName()
        self.hasData=False
        self.df=None
        self.channelList=None

    def loadData(self,filename,timeColumnName='DATETIME',
                 timeshiftHour=0,timeshiftSecond=0,savePickle=False):
        pickleFileName = filename + '.p'
        if os.path.isfile(pickleFileName) == False:
            print('pickle file does not exist, load from csv')

            dataraw = pd.read_csv(filename, skiprows=[1])
            #    dataraw
            # % filter data and change index to timestamp
            timestamp = pd.to_datetime(dataraw[timeColumnName],infer_datetime_format=True)
            timestamp = timestamp - pd.offsets.Second(timeshiftSecond) - pd.offsets.Hour(timeshiftHour)
            dataraw[timeColumnName] = timestamp
            data = dataraw.set_index(timeColumnName)
            del dataraw
            pickleFileName = filename + '.p'
            t0 = time()
            if savePickle:
                with open(pickleFileName, 'wb') as file:
                    pickle.dump(data, file, protocol=4)
                print('done writing pickle in %.2fs.' % (time() - t0))
        else:
            print('load from pickle file')
            t0 = time()
            with open(pickleFileName, 'rb') as file:
                data = pickle.load(file)
            print('done reading pickle in %.2fs.' % (time() - t0))
        self.hasData=True
        self.df=data
        self.channelList = [s for s in self.df.columns]

    def setData(self,dataframe):
        self.df=dataframe
        self.hasData=True
        self.channelList = [s for s in self.df.columns]

    def getStatsData(self, depthArray, depthSeriesName, channelList=None, stats=None):
        """return a new dataframe with statistics"""
        if stats is None:
            stats = ['mean']
        if channelList is None:
            channelListToUse=self.channelList
        else:
            channelListToUse=channelList
            channelToRemove=[]
            for c in channelListToUse:
                print(c)
                if c not in self.channelList:
                    channelToRemove.append(c)
            for c in channelToRemove:
                channelListToUse.remove(c)
                    # print("removing  " + c)
                    # channelListToUse.remove(c)

        # datetime=self.df.index
        # self.df['DateTime']
        datetimeList=self.calcTimeByDepth(depthArray, self.df[depthSeriesName])

        # data,names = self.calcStatMultiChannel(depthArray,self.df[depthSeriesName],
        #                                        channelList=channelListToUse,stats=['mean','max','min'])

        data,names = self.calcStatMultiChannel_fast(depthArray,self.df[depthSeriesName],
                                               channelList=channelListToUse,stats=stats)

        df_stat = pd.DataFrame(data=data,columns=names,index=datetimeList)

        return df_stat

    def calcStatMultiChannel_fast(self, depthArray, depthSeries, channelList, stats):
        # result  = list()
        if type(stats)!=list:
            stats=[stats]
        nstat=len(stats)
        nchannel = len(channelList)
        ndepth = len(depthArray)

        # calculate data first
        outArray=np.zeros((ndepth-1,nstat*nchannel))*np.nan
        # loop through depth array to calculate statistics for each interval
        for i in range(0,ndepth-1):
            perc=i/ndepth*100
            if perc%5==0:
                print('%2.2f percent finished'%perc)
            mask=(depthSeries>= depthArray[i]) & (depthSeries< depthArray[i+1])
            datacut=self.df.loc[mask]

            # for ch in channelList:
            for ic in range(len(channelList)):
                istat=0
                for stat in stats:
                    subset = np.array(datacut[channelList[ic]])
                    outArray[i,ic*nstat+istat]=self.calcStat(subset,stat)
                    istat+=1


        # then do channel name
        outChannelNames=list()
        for ic in range(len(channelList)):
            # istat = 0
            for stat in stats:
                outChannelNames.append(channelList[ic]+'-'+stat)
                # istat += 1
        return outArray,outChannelNames



    def calcStatMultiChannel(self,depthArray,depthSeries,channelList,stats):
        """Calculate multiple statistics of multiple channels """
        result  = list()
        outChannelNames=list()
        if type(stats)!=list:
            stats=[stats]
        for s in channelList:
            print('calculating statistics for channel '+ s)
            for stat in stats:
                print(stat)
                outArray=self.calcStatByDepth(depthArray,depthSeries,self.df[s],stat)
                result.append(outArray)
                outChannelNames.append(s+'-'+stat)
                data = np.array(result).transpose()
        return data,outChannelNames


    def calcStat(self,input,stat):
        if input.size > 2:
            if stat == 'mean':
                # assert isinstance(subset, object)
                return np.nanmean(input)
            elif stat == 'median':
                return np.nanmedian(input)
            elif stat == 'max':
                return np.nanmax(input)
            elif stat == 'min':
                return np.nanmin(input)
            elif stat=='sum':
                return np.nansum(input)
        return np.nan

    def calcStatByDepth(self,depthArray,depthSeries,inputSeries,stat='mean'):
        """Calculate certain statistics of a single channel """
        outArray=np.zeros(len(depthArray)-1)*np.nan
        # loop through depth array to calculate statistics for each interval
        ndepth = len(depthArray)
        for i in range(0,ndepth-1):
            mask=(depthSeries>= depthArray[i]) & (depthSeries< depthArray[i+1])
            subset=np.array(inputSeries.loc[mask])
            if subset.size>2:
                if stat == 'mean':
                    # assert isinstance(subset, object)
                    outArray[i]=np.nanmean(subset)
                elif stat == 'median':
                    outArray[i] = np.nanmedian(subset)
                elif stat == 'max':
                    outArray[i] = np.nanmax(subset)
                elif stat == 'min':
                    outArray[i] = np.nanmin(subset)
        return outArray

    def calcTimeByDepth(self,depthArray,depthSeries,timeColumnName='DateTime'):
        df_tmp=pd.DataFrame(index=self.df.index)
        df_tmp['DateTime']=self.df.index

        # outArray = np.zeros(len(depthArray) - 1) * np.nan
        outList=[]
        # loop through depth array to calculate statistics for each interval
        ndepth = len(depthArray)
        for i in range(0,ndepth-1):
            mask=(depthSeries>= depthArray[i]) & (depthSeries< depthArray[i+1])
            subset=df_tmp['DateTime'].loc[mask]
            outList.append(self.meanDateTime(subset))
            # outList.append(np.min(subset))

        return outList

    def calcTimeByDepth_new(self,depthArray,depthSeries,timeColumnName='DateTime'):
        df_tmp=pd.DataFrame(index=self.df.index)
        df_tmp['DateTime']=self.df.index

        # outArray = np.zeros(len(depthArray) - 1) * np.nan
        outList=[]
        # loop through depth array to calculate statistics for each interval
        ndepth = len(depthArray)
        for i in range(0,ndepth-1):
            mask=(depthSeries>= depthArray[i]) & (depthSeries< depthArray[i+1])
            subset=df_tmp['DateTime'].loc[mask]
            outList.append(self.meanDateTime(subset))
            # outList.append(np.min(subset))

        return outList



    @staticmethod
    def meanDateTime(datetimeSeries):
        datetimeStart=datetimeSeries.min()
        datetimeEnd=datetimeSeries.max()
        return datetimeStart+(datetimeEnd-datetimeStart)/2

    # TODO:
    def findHeelDepth(self,svymdArray,svyincArray,incLimit=87):
        # sort survey data according to depth
        # idx=np.argsort(svymdArray)
        # svyincArray=svyincArray[idx]
        # svymdArray=svymdArray[idx]

        mask= (svyincArray>=incLimit)
        if len(svymdArray.loc[mask])>0:
            heelDepth=svymdArray.loc[mask][0]
        else:
            heelDepth=np.NaN
        return heelDepth

    def calcDistanceFromHeel(self,mdArray,heelDepth):
        return mdArray-heelDepth