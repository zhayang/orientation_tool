#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:58:15 2018
Test class to generate batch training data

@author: zhay
"""

#%% 

import deep_util
import json
#--------------------Well specific info---------------------------------
# data files 
datafile='/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv'
outFileName='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/test.p'

##bitter owen colin A
emsdatafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/EMS/BITTERLY_OWENS_COLIN_A_ULW#1_BHA02R01_EMS01_SNEMSIB38_MEMORYMERGED.csv'
emshifidatafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/EMS/bha02_merge_hifi.csv'
timeshiftHour=0
timeshiftSecond=204
#--------------------below should not change for each well--------------------------------
## processing parameters
depthMin=5000/3.28;
depthMax=7000/3.28;
fs=100
ntSample=int(60*fs)
ntSampleStep=ntSample*0.5
ntSegment=1024
ntSegOverlap=ntSegment*0.75


## channels 

inputChannel=['01_GLOBAL_PASON_TTS-STRQ', 
              '01_GLOBAL_PASON_TTS-TENSION', 
              '01_GLOBAL_PASON_TTS-SRPM']


auxChannelList=['01_GLOBAL_GENERIC_SURFACE-SWOB', 
                '01_GLOBAL_GENERIC_SURFACE-copRigState',
                '01_GLOBAL_GENERIC_SURFACE-BIT_DEPTH',
                '01_GLOBAL_PASON_TTS-SRPM']

targetChannel=['EMS_LLAT_MAX(G)']

#----------Process and save----------

dataSelector=deep_util.DataSelector()
dataSelector.setProcessingParam(depthMin=depthMin,depthMax=depthMax,ntSample=ntSample,
                                ntSampleStep=ntSampleStep,ntSegment=ntSegment,ntSegOverlap=ntSegOverlap)
dataSelector.setDHDataSource(name=emsdatafile,timeColumnName='TIME(datetime)',
                             timeShiftHour=timeshiftHour,timeShiftSecond=timeshiftSecond)

dataSelector.loadData(datafile)
dataSelector.loadDHData()

#result=dataSelector.preprocess(inputChannelList=inputChannel,targetChannelList=targetChannel)

auxDataDict=dataSelector.makeAuxData(auxChannelList)
result=dataSelector.makedata(inputChannelList=inputChannel,targetChannelList=targetChannel,auxChannelList=auxChannelList)


dataSelector.saveToFile(outFileName,result)

#%%  test multiple wells from multiple files

wellList=[1,2]
outfileList=['/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/test1.p',
             '/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/test2.p']
dataSelector=deep_util.DataSelector()
dataSelector.setProcessingParam(depthMin=depthMin,depthMax=depthMax,ntSample=ntSample,
                                ntSampleStep=ntSampleStep,ntSegment=ntSegment,ntSegOverlap=ntSegOverlap)
for outfile in outfileList:
#   get data file info 
    dataSelector.setDHDataSource(name=emsdatafile,timeColumnName='TIME(datetime)',
                             timeShiftHour=timeshiftHour,timeShiftSecond=timeshiftSecond)
#   load data
    dataSelector.loadData(datafile)
    dataSelector.loadDHData()
    
#   generate samples
    result=dataSelector.makedata(inputChannelList=inputChannel,targetChannelList=targetChannel)

    dataSelector.saveToFile(outfile,result)
    print('writing outfile ' + outfile)
    dataSelector.resetData()
#   merge dataset

#%% test json reading
cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/test.json'
with open(cfgfile) as f:
    jdata = json.load(f)
print(jdata)

    #%% test processing multiple wells from configuration .json files
import json
import deep_util
#import pdb
#cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/test.json'
#cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/rr_process_10s.json'
#cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/rr_process_30s.json'
#cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/rr_process_drpm_30s.json'
#cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/rr_process_30s_noNorm.json'
#cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/colinA_process_30s.json'
cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/rr_process_30s_axial.json'

#cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/rr_process.json'
#cfgfile='/home/zhay/DEEP/machineLearning/vibrationDetector/testClass/colinA_process.json'

with open(cfgfile) as f:
    jdata = json.load(f)
#print(jdata)

depthMin=5000/3.28;
depthMax=18000/3.28;
fs=100
ntSample=int(30*fs)
ntSampleStep=ntSample*1
ntSegment=1024
ntSegOverlap=ntSegment*0.75


inputChannels=jdata['inputChannels']
auxChannels=jdata['auxChannels']
targetChannels=jdata['targetChannels']
#numwell = len(jdata['wells'])
dataSelector=deep_util.DataSelector()   
dataSelector.setProcessingParam(depthMin=depthMin,depthMax=depthMax,ntSample=ntSample,
                                ntSampleStep=ntSampleStep,ntSegment=ntSegment,ntSegOverlap=ntSegOverlap)
for well in jdata['wells']:
    print('------------Processing  '+ well['name']+'------------')
    
    dataSelector.setDHDataSource(name=well['dhDataFile'],timeColumnName='TIME(datetime)',
                             timeShiftHour=well['timeshiftHour'],timeShiftSecond=well['timeshiftSecond'])
#   load data
    dataSelector.loadData(well['dataFile'])
    dataSelector.loadDHData()
    
#   generate samples
#    pdb.set_trace()
    result=dataSelector.makedata(inputChannelList=inputChannels,
                                 targetChannelList=targetChannels,
                                 auxChannelList=auxChannels,normalize=True)

    dataSelector.saveToFile(well['outFile'],result)
    print('writing outfile ' + well['outFile'])
#    dataSelector.resetData()