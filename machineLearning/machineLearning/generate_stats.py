#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 15:11:32 2018

@author: zhay
"""

import sys
import json
import argparse
import numpy as np
sys.path.append('/home/zhay/DEEP/python/')
sys.path.append('/home/zhay/DEEP/python/lib/')
sys.path.append('/home/zhay/DEEP/python/machineLearning/')

#import deep_util
#import matplotlib.pyplot as plt
#from stickslipDetector import StickSlipDetector
#from DrillingDataProcessor import StickSlipProcessor
from drillingDataFrame import DrillingDataFrame
from time import time
import deep_util

#print('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))

#load arguments
if len(sys.argv) > 1:
    args = sys.argv[1:]
else:
    args = ''
parser = argparse.ArgumentParser(description='Machine learning batch processing on DEEP data')
#parser.add_argument('-dt', dest='dt', type=float, default=0.01)
parser.add_argument('-inFile', dest='inFile', type=str, required=True)
parser.add_argument('-outFile', dest='outFile', type=str, required=True)
parser.add_argument('-cfgFile', dest='cfgFile', type=str, required=True)
parser.add_argument('-wellName', dest='wellName', type=str, required=True)

options = parser.parse_args(args)

#read options from json
with open(options.cfgFile) as f:
    jdata = json.load(f)

dt = jdata['dt']
depthMin = jdata['depthMin']
depthMax = jdata['depthMax']
depthInterval = jdata['depthInterval']
depthChannel = jdata['depthChannel']

#--------------------------------------
#read data
deepdata=deep_util.loadDEEPData(options.inFile,timeshiftHour=0,timeshiftSecond=0,savePickle=False)
mask= (deepdata[depthChannel]>depthMin) & (deepdata[depthChannel]< depthMax)
deepdatacut = deepdata.loc[mask]


#generate statistics report 
ddf=DrillingDataFrame(dt=dt)
ddf.setData(deepdatacut)
depthArray = np.linspace(depthMin,depthMax,num=int(np.floor((depthMax-depthMin)/depthInterval)))
t0 = time()
#pdb.set
df_stat_raw=ddf.getStatsData(depthArray,channelList=jdata['channelList'],depthSeriesName=depthChannel,stats=jdata['stats'])
df_stat=df_stat_raw.dropna(how='all')
df_stat.index.name='DATETIME'
df_stat['Wellname'] = options.wellName

print('done in %.2fs.' % (time() - t0))

#save as csv 
df_stat.to_csv(options.outFile)
