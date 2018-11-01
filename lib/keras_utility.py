#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 08:25:08 2017

@author: zhay
"""

import numpy as np
#from scipy import signal
from scipy import interpolate
#from spectral import calc_coherence,calc_spec
import matplotlib.pyplot as plt
import pickle
#%%
import keras.backend as K
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def cut_to_batch(features, labels, batchSize):

    # Needed for stateful training: make sure num samples is commensurate with batch size
    batchMultiple = np.int(np.float(features.shape[0])/batchSize) * batchSize
    features = features[:batchMultiple]
    if labels is not None:
        labels = labels[:batchMultiple]

    return features, labels

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
#    return f1_score(y_true, y_pred, beta=1)
    p=precision(y_true, y_pred)
    r=recall(y_true, y_pred)
    return 2 * (p*r) / (p+r)

def binary_xentropy(y_true, y_pred):
    result = []
    for i in range(len(y_pred)):
        y_true[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_true[i]]
        y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_pred[i]]
#        result.append(-np.mean([y_true[i][j] * math.log(y_pred[i][j]) + (1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))
        result.append([y_true[i][j] * np.log(y_pred[i][j]) + (1 - y_true[i][j]) * np.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))])

    return -1*np.mean(result)


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
    

#% augmenting data by squeezing and streching
def augmentData(X,y,multiplyRatio,stretchRatios=[0.8, 1.2],addNoise=False,noiseAmp=0.1,mergeOutput=False):
    np.random.seed(1)
#    number of data to generate
    ndataTrue=np.int(np.sum(y))
    ndataAdd = np.int(ndataTrue*multiplyRatio)
#    separate true vs false
#    Xfalse=X[y==0]
    maskTrue=np.reshape(y,(-1,))==1
    Xtrue=X[maskTrue]
    
#    loop through each true sample
    XGen=np.zeros((ndataAdd,X.shape[1],X.shape[2]))
    yGen=np.zeros((ndataAdd,1))
    vec_ratio=np.zeros(ndataAdd)
    j=0    
    for isample in range(0,ndataTrue):
        
        xOrig=Xtrue[isample,:,:]
#        generate multiple synthetic data
        for istretch in range(0,np.int(multiplyRatio)):            
#            print('processing window %3.0f'%iwin)
            ratio=np.random.random()*(stretchRatios[1]-stretchRatios[0])+stretchRatios[0]
            
            if ratio==1:
                ratio=ratio+0.1
                print('ratio = %3.3f'%ratio)

            if addNoise==True:
#                XGen[j,:,:]=stretchTimeSeries(addRandomNoise(xOrig,noiseAmp),ratio) # add noise then stretch
                XGen[j,:,:]=addRandomNoise(stretchTimeSeries(xOrig,ratio),noiseAmp) # add noise after stretch

            else:
                XGen[j,:,:]=stretchTimeSeries(xOrig,ratio)

#            XGen[j,:,:]=stretchTimeSeries(xOrig,ratio)
            yGen[j]=1
            vec_ratio[j]=ratio
            j=j+1
    if mergeOutput: 
        Xout=np.concatenate((X,XGen),axis=0)
        yout=np.concatenate((y,yGen),axis=0)
        return Xout,yout,vec_ratio
    else:
        return XGen, yGen,vec_ratio

def addRandomNoise(x,noiseAmp=0.1):
#    nt=x.shape[0]
#    nchannel=x.shape[1]
    noise=noiseAmp*np.random.rand(x.shape[0],x.shape[1])
    x_noisy=noise+x
    return x_noisy

#def addRandomNoise(X,)
def stretchTimeSeries(x,ratio):
    nt=x.shape[0]
    nchannel=(x.shape[1])
    vec_t0=np.arange(0,nt)
    vec_t1=vec_t0*ratio
    
    ntcut=np.int(nt/ratio)
#    if ratio>1:
    if np.int(nt*(1-1/ratio))>0:
#        print('idxStart = %3.3f'%np.int(nt*(1-1/ratio)))
        idxStart=np.random.randint(0,np.int(nt*(1-1/ratio)))
    else:
        idxStart=0
    
    idxEnd=idxStart+ntcut
    xcut=x[idxStart:idxEnd,:]

    
#    xinterp=np.interp(vec_t0,vec_t1[0:ntcut],xcut)
    f=interpolate.interp1d(vec_t1[0:ntcut],xcut,axis=0,bounds_error=False,fill_value=np.zeros(nchannel))
    xinterp=f(vec_t0)
    return xinterp
    

def plot_trainingHistory(History):
    # Training history
    fig=plt.figure(figsize=[12,10],dpi=150)
    # fig.set_size_inches(12,10)
    ax1=plt.subplot(311)
    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # fig=plt.figure(figsize=[12,10])

    ax2=plt.subplot(312)
    # fig.set_size_inches(12,10)

    # summarize history for loss
    plt.plot(History.history['precision'])
    plt.plot(History.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    #plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # plt.set_size_inches(12,10)

    # fig=plt.figure(dpi=150)
    ax3=plt.subplot(313)
    # summarize history for loss
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
#     return fig


def normalizeData(dataIn,removeMean=True,normalizeAmp=True,epsilon=0.0001,axis=1):
    dataOut=dataIn
    if(removeMean):
        dataOut=dataOut-np.nanmean(dataIn,axis=axis,keepdims=True)
    if(normalizeAmp):
        dataOut=dataOut/(np.nanstd(dataIn,axis=axis,keepdims=True)+epsilon)
    #    normalize data for training
    return dataOut
    