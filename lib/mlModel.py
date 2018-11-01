# coding=utf-8
import numpy as np
import scipy as sp
import pandas as pd
from abc import ABC, abstractmethod


class AbstractMLOperator(ABC):
    """Python ABC class"""
    def __init__(self,dt):
        self.dt = dt
        self.nchannel=None
        self.ntStep=None
        # super.__init__(self)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def getData(self):
        print('This is the abstract class')


class MLOperator(object):
    """Base class"""
    # def __init__(self):
    def __init__(self,dt):
        self.dt = dt
        self.nchannel=None
        self.ntStep=None


    def initialize(self):
        """set processing paramters"""
        print('do some processing')
        raise NotImplementedError("This is a required method, do something")

    def getData(self):
        """get data as numpy arrays"""

        raise NotImplementedError("This is a required method")

    def process(self):
        """do some processing"""
        raise NotImplementedError("This is a required method")


    def getPrediction(self):
        """get back result"""
        raise NotImplementedError("This is a required method")

class DummyMLOperator2(AbstractMLOperator):
    def initialize(self,ntStep=1,nchannel=0,channelList=[]):
        self.ntStep=ntStep
        self.nchannel=nchannel
        self.channelList=channelList
        print('Initialize operator')
        print('Using channels:')
        print(channelList)

    # def getData(self):
    #     print('This is the child class')
class DummyMLOperator(MLOperator):
    def __init__(self,dt):
        self.dt = dt
        self.nchannel=None
        self.ntStep=None
        self.channelList=None
        self.torqueArray=None
        self.rpmArray=None
        self.result=None

    def initialize(self,ntStep=1,nchannel=0,channelList=[]):
        self.ntStep=ntStep
        self.nchannel=nchannel
        self.channelList=channelList
        print('Initialize operator')
        print('Using channels:')
        print(channelList)
    # def initialize(self):

    def getData(self, dataFrame):
        dataList=[np.array(dataFrame[ch]) for ch in self.channelList]
        return dataList

    # call for every sample
    def process(self,dataList):
        assert len(dataList)==self.nchannel
        return dataList[0]*dataList[1]


def main():
        import matplotlib.pyplot as plt
        op=DummyMLOperator(1)
        ntstep=10
        op.initialize(ntStep=ntstep,nchannel=2,channelList=['rpm','torque'])
        data=np.random.random((300,2))
        df=pd.DataFrame(data=data,columns=['rpm','torque'])
        nwin=10
        istart=0
        iend=ntstep
        while iend<df.shape[0]:
            data=op.getData(df.loc[istart:iend])
            result=op.process(data)
            print(result)
            istart+=ntstep
            iend=istart+ntstep

if __name__ =='__main__':
    main()
