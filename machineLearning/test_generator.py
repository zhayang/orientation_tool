#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:22:19 2018

@author: zhay
"""
#%%
#test data generator for training 
import numpy as np

from dataGenerator import BalancedDataGenerator_binary


#from dataGenerator import BalancedDataGenerator_binary
npos = 100
nneg = 1000
feature_neg = np.random.rand(nneg, 30, 3)
feature_pos = np.random.rand(npos, 30, 3)
label_neg = np.zeros((nneg,30,2))
label_pos = np.random.rand(npos,30,2)

# label_neg = np.zeros((nneg,))
# label_pos = np.ones((npos,))

features = np.concatenate((feature_neg,feature_pos),axis=0)
labels = np.concatenate((label_neg,label_pos),axis=0)
maxClass = np.max(np.argmax(labels, axis=2), axis=1)


gen = BalancedDataGenerator_binary(features,labels,scalerLabel=maxClass,
                                   batch_size=32,shuffle=True,classProbability=[1,1])

nbatch = gen.__len__()

for epoch in range(10): 
    gen.on_epoch_end()
    print('epoch'+ str(epoch))
    for i in range(nbatch):
        xb,yb = gen.__getitem__(i)
        
        maxClassBatch  =np.max(np.argmax(yb, axis=2), axis=1)
        print(len(xb))
#        print(yb)
        print(gen.ind_pos_augment[:5])
        print('percentage of positive sample is '+ str(sum(maxClassBatch)/len(yb)))
    
    