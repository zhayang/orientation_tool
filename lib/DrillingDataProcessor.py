# coding=utf-8
"""
A csv file handler class for batch processing Drilling data with ML/analytics operator
"""

from deep_util import loadDEEPData
import pandas as pd
import numpy as np
import pickle
import os
import json
from stickslipDetector import StickSlipDetector
class DataFrameHandler:
    """ handles reading and writing of deep format data from different format"""
    def __init__(self,inFile=None, outFile=None,dt=1.0):
        self.channelList=None
        self.options = None
        self.inFile=inFile
        self.outFile=outFile
        self.outChannelList=None
        self.dt=dt

    def loadCSVData(self,filename,timeColumnName='DATETIME',
                 timeshiftHour=0,timeshiftSecond=0,savePickle=False):
        self.inFile=filename
        pickleFileName = filename + '.p'
        if os.path.isfile(pickleFileName) == False:
            print('pickle file does not exist, load from csv')

            dataraw = pd.read_csv(filename, skiprows=[1])
            #    dataraw
            # % filter data and change index to timestamp
            timestamp = pd.to_datetime(dataraw[timeColumnName],infer_datetime_format=True,errors='coerce')
            timestamp = timestamp - pd.offsets.Second(timeshiftSecond) - pd.offsets.Hour(timeshiftHour)
            dataraw[timeColumnName] = timestamp
            data = dataraw.set_index(timeColumnName)
            del dataraw
            pickleFileName = filename + '.p'
            if savePickle:
                with open(pickleFileName, 'wb') as file:
                    pickle.dump(data, file, protocol=4)
        else:
            print('load from pickle file')
            with open(pickleFileName, 'rb') as file:
                data = pickle.load(file)

        # self.df = data
        self.channelList = [s for s in data.columns]
        return data

    def writeCSVData(self,outData,outFile):
        if(outData is not None) and (type(outData)==pd.core.frame.DataFrame):
            outData.to_csv(outFile)
        else:
            raise TypeError("data is not a DataFrame")

    def process(self):
        """do some processing"""
        raise NotImplementedError("This method has not been implemented")

class StickSlipProcessor(DataFrameHandler):
    """A class that applies stickslip operator to a csv file and output the results"""

    def __init__(self,dt=1.0,infile=None,outfile=None,cfgfile=None):

        super(StickSlipProcessor,self).__init__(inFile=infile,outFile=outfile,dt=dt)
        self._infile = infile
        self._outfile = outfile
        self._cfgfile = cfgfile
        self._options=None
        self._data = None
        self._target = None
        self.loadOptions()
        self._mloperator = StickSlipDetector(dt)



    def loadOptions(self):
        """ initilaize options from the cfg/json file"""
        with open(self._cfgfile) as f:
            jdata = json.load(f)
        self._options=jdata
        # self.ntStep=jdata['ntStep']
        # self.nchannel=jdata['nchannel']
        # self.channelList=jdata['channelList']
        # self.modelFile=jdata['modelFile']
        # self.isNormlize=jdata['isNormlize']
        # self.batchSize=jdata['batchSize']
        # self.rpmlimit=jdata['rpmlimit']
        # self.timeColumnName=jdata['timeColumnName']
        # self.savePickle=jdata['savePickle']
        # self.depthMin=jdata['depthMin']
        # self.depthMax=jdata['depthMax']
        return None

    def initialize(self):

        # load data from file
        dataraw=self.loadCSVData(filename=self._infile,
                         timeColumnName=self._options['timeColumnName'],
                         savePickle=self._options['savePickle'])


        # select data from depth interval
        holedepth=dataraw[self._options['holedepthName']]
        mask = (holedepth > self._options['depthMin']) & (holedepth < self._options['depthMax'])
        self._data = dataraw.loc[mask]

        # initalize ml operator
        self._mloperator.initialize(ntStep=self._options['ntStep'],
                                    nchannel=self._options['nchannel'],
                                    modelFile=self._options['modelFile'],
                                    isNormlize=self._options['isNormalize'],
                                    batchSize=self._options['batchSize'],
                                    rpmlimit=self._options['rpmLimit'])

        self._target = pd.Series(index=self._data.index)


    def process(self):
        """ loop through each frame/batch to perform the calculation"""
        ntTot = self._data.shape[0]
        ntStep= self._options['ntStep']
        batchSize= self._options['batchSize']

        istart=0
        iend=ntStep*batchSize
        nwin = ntTot // ntStep
        nextLevel = 1

        while iend < ntTot:
            #    pdb.set_trace()
            dataList = self._mloperator.getData(self._data[istart:iend],channelList=self._options['channelList'])
            self._target[istart:iend] = self._mloperator.process(dataList)
            istart += ntStep * batchSize
            iend = istart + ntStep * batchSize
            perc = iend / ntTot * 100
            if perc > nextLevel:
                print('%2.0f percent finished' % perc)
                nextLevel = np.floor(perc)+1

        dfOut=self._data.copy()
        dfOut[self._options['targetColName']] =self._target
        return dfOut


def main():
    # inFileName = '/home/zhay/DEEP/deep_data/BLUEBERRY_A13-22/ExportData_ASCII_ByTime_Precision867.csv'
    # outFileName = inFileName+'.copy'
    # # dfh=DataFrameHandler(dt=0.01)
    # # #
    # # dfh.loadCSVData(filename=inFileName,timeColumnName='DateTime')
    # # dfh.writeCSVData(dfh.df,outFileName)

    datafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv'  #
    outfile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/test_prediction.csv'  #
    cfgfile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/test_stickslip_proceesing.json'

    op = StickSlipProcessor(dt=0.01, infile=datafile, outfile=outfile, cfgfile=cfgfile)
    # op.loadOptions()
    op.initialize()
    dfOut = op.process()
    op.writeCSVData(dfOut, outfile)

if __name__ =='__main__':
    main()


