# coding=utf-8
import numpy as np
# import scipy as sp
# import pandas as pd
# from abc import ABC, abstractmethod
# from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv1D,Conv2D, MaxPooling2D,MaxPooling1D
# from keras.losses import binary_crossentropy, categorical_crossentropy
#
# #import os
# import pickle
from keras.models import load_model, save_model
from keras.optimizers import Adam
from mlModel import MLOperator

class StickSlipDetector(MLOperator):
    """A ML operator that classify surface channels into severe stick-slip (1) or normal (0)
    Input : N by nt np arrays
    Output: 1 by nt np array """

    def __init__(self,dt):
        self.dt = dt
        self.nchannel=None
        self.ntStep=None
        self.channelList=None
        self.torqueArray=None
        self.rpmArray=None
        self.result=None
        self.kerasModel=None
        self.epsilon=0.0001
        self.batchSize=None
        self.isNormalize=None
        self.RPMLimit=None
        self.returnProbability=None


    def initialize(self,ntStep,nchannel,modelFile,
                   isNormlize=True,batchSize=1,rpmlimit=40,returnProb=True,
                   optimizer=Adam(lr=0.001, decay=1e-6),
                   loss='binary_crossentropy'):
        self.ntStep=ntStep
        self.nchannel=nchannel
        self.kerasModel = load_model(modelFile,compile=False)
        self.kerasModel.compile(loss=loss,optimizer=optimizer)
        self.batchSize=batchSize
        self.isNormalize=isNormlize
        self.RPMLimit=rpmlimit
        self.returnProbability=returnProb
        print('Initialize operator')


    def getData(self, dataFrame, channelList):
        dataList=[np.array(dataFrame[ch]) for ch in channelList]
        return dataList

    # call for every sample
    def process(self,dataList):
        # standardiation
        
        dataInReshape=self.reshapeBatch(dataList[:3])
        
        dataIn = self.standardizeBatch(dataInReshape)
        # np.zeros((self.batchSize,self.ntStep,self.nchannel))


        isRotating = np.where(dataList[3]>=self.RPMLimit,1,0)
        # print(sum(isRotating))

        # processing
        y_prob = self.kerasModel.predict(dataIn)
        y_label= np.round(y_prob)

        # reshape to ntstep x batchsize
        ylabelReshape=np.ones((self.batchSize,self.ntStep))*y_prob
        # then to 1 x ntstep x batchsize
        ylabelFlattern = np.reshape(ylabelReshape,(self.batchSize*self.ntStep,))
        # return array of the same size
        result=ylabelFlattern*isRotating
        if ~self.returnProbability:
            result = np.round(result)
        return result

    def reshapeBatch(self,dataList):
        # reshape data into nt*batchsize by nchannel
        dataArray = np.transpose(np.asarray(dataList))
        # reshape into batchsize by nt by nchannel
        result=np.reshape(dataArray,(self.batchSize,self.ntStep,self.nchannel))
        # result = np.zeros((self.batchSize,self.ntStep,self.nchannel))

        return result

    def standardizeBatch(self,X):
        """standardize input data in batch
        shape of X is (batchsize,nt,nchannel)"""
        Y = np.zeros(X.shape)
        for ic in range(self.nchannel):
            for isample in range(self.batchSize):
                if self.isNormalize:
                    Y[isample, :, ic] = self.standardize(X[isample,:,ic])
                else:
                    Y[isample, :, ic] = X[isample,:,ic]
        return Y


    def standardize(self,x):
        return (x-np.nanmean(x))/(np.nanstd(x)+self.epsilon)


def main():
    import deep_util
    import matplotlib.pyplot as plt

    # load data
    # load data
    # datafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv'  #
    datafile = '/home/zhay/DEEP/deep_data/test_ss/BLUEBERRY_A13-22/BLUEBERRY_A13-22_merge_cut_100hz_stickslip.csv'  #

    data = deep_util.loadDEEPData(datafile, savePickle=True)

    depthMin = 6000 / 3.28;
    depthMax = 16000 / 3.28;

    if ('01_Global-Generic_Surface-HOLEDEPTH' in data.columns):
        holedepthName = '01_Global-Generic_Surface-HOLEDEPTH'
        bitdepthName = '01_Global-Generic_Surface-BIT_DEPTH'
    else:
        holedepthName = '01_GLOBAL_GENERIC_SURFACE-HOLEDEPTH'
        bitdepthName = '01_GLOBAL_GENERIC_SURFACE-BIT_DEPTH'

    mask = (data[holedepthName] > depthMin) & (data[holedepthName] < depthMax)
    deepDF = data.loc[mask]



    # set model parameters
    model_name = '/home/zhay/DEEP/machineLearning/stickslipDetector/recurrent_model/stickslipDetector_conv_lstm16_merge.h5'
    ntStep = 3000
    ntOverlap = 0
    # channelList=['01_GLOBAL_PASON_TTS-STRQ',
    #              '01_GLOBAL_GENERIC_SURFACE-SWOB',
    #              '01_GLOBAL_PASON_TTSV-DRPM',
    #              '01_GLOBAL_GENERIC_SURFACE-SRPM']
    channelList=["01_GLOBAL_NOV_STRINGSENSE-STRQ",
             "01_GLOBAL_GENERIC_SURFACE-SWOB",
             "01_GLOBAL_NOV_STRINGSENSEV-DRPM",
             "01_GLOBAL_NOV_STRINGSENSE-SRPM"]

    # create operator
    op = StickSlipDetector(dt=0.01)

    # op=DummyMLOperator(dt=0.01)

    op.initialize(ntStep=ntStep,rpmlimit=10,nchannel=3,modelFile=model_name,isNormlize=True)
    ntTot = deepDF.shape[0]
    nwin = ntTot // ntStep

    target = np.zeros(deepDF.shape[0])
    istart = 0
    iend = ntStep
    # simulate streaming environment
    while iend < 300000:
        # while iend < deepDF.shape[0]:

        dataList = op.getData(deepDF[istart:iend],channelList=channelList)
        target[istart:iend] = op.process(dataList)
        istart += ntStep
        iend = istart + ntStep


    plt.figure()
    plt.plot(target)
    plt.show()
if __name__ =='__main__':
    main()