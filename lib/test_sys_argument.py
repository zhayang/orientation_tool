#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:31:40 2018

@author: zhay
"""

#test system argument parsing

import sys
import argparse

#%%
print('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

if len(sys.argv) > 1:
    args = sys.argv[1:]
else:
    args = ''
    
parser = argparse.ArgumentParser(description='Machine learning batch processing on DEEP data')
parser.add_argument('-dt', dest='dt', type=float, default=0.01)
parser.add_argument('-inFile', dest='inFile', type=str, required=True)
parser.add_argument('-outFile', dest='outFile', type=str, required=True)
parser.add_argument('-cfgFile', dest='cfgFile', type=str, required=True)
options = parser.parse_args(args)

print('dt is '+ str(options.dt))
print('input file is '+ options.inFile)
#print('dt is '+ options.dt)
