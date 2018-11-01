#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:40:33 2017

Spectral signal procecesing module

@author: zhay
"""
import scipy as sp
#import pybrain as pb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
from scipy.fftpack import fft,ifft,fftfreq
from scipy.signal import blackman,hamming,hanning
from scipy.signal import butter, lfilter, filtfilt
import pdb
#%%Spectral deconvolution
#deconvolve divisor from signal
def specDecon(dt,signal,filt,epsilon=1e-6):
    specS=fft(signal)
    specD=fft(filt)
    nfSignal=len(specS)
    nfFilter=len(filt)
    
#%% Spectrogram 
def calc_spectrogram(y,fs,ntSegment,ntOverlap):
#    call scipy spectogram
    vf,_,Sxx=sp.signal.spectrogram(y,fs=fs,nperseg=ntSegment,noverlap=ntOverlap)
    return vf,Sxx
#%%remove correlated signal between 2 channels 
#    time domain 
#    y1: source , y2:response
def remove_coh_signal2(y1,y2,dt,ntwin,fmin,fmax,overlapPerc=0,detrend=True,type='phase'):
    """
    Remove part of signal 2 (response) that is coherent with signal 1 (source)
    Args:
        y1: signal
        y2: targer response that is partially coherent with signal
    Returns:
        y2_1
    """
    fs=1/dt
    if fmax>=fs/2:
        fmax=fs/2-0.01
        
#    determine total number of points and frequency vector
    ntTot=y2.shape[0]
    y2_1=y2.copy()
    
    nfft=np.power(2,np.int(np.log2(ntwin))+1)
    nt_overlap= int(nfft*overlapPerc)
#    nwin_initial=(ntTot-nt_overlap_initial)/(nfft-nt_overlap_initial)
#    nt_overlap=np.int((nwin_initial*nfft-ntTot)/(nwin_initial-1))
#    print(nfft)
#    print(nt_overlap)
    vec_f=np.linspace(0,1/2/dt,np.int(nfft//2)+1)
#    vec_f=fftfreq(nfft,dt)
    nf=np.int(nfft/2)+1;

#    initiate stacking
    c11_stack=np.zeros(nf)
    c22_stack=np.zeros(nf)
    c12_stack=np.zeros(nf)

    itStart=0
    itEnd=nfft
    nwin_stack=0
#   loop through for the first time    
    isLast=False
    isEnd=False
    while not isEnd:
        if isLast:
            isEnd=True
            
#        print('end index')        
#        print(itEnd)
        y1_win=y1[itStart:itEnd];#%*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        y2_win=y2[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        if np.isinf(y1_win).any() or np.isinf(y2_win).any() or np.isnan(y1_win).any() or np.isnan(y2_win).any():
            continue
        y1_win=signal.detrend(y1_win);
        y2_win=signal.detrend(y2_win);
        window=hamming(len(y1_win))
        
        y1_win=y1_win*window
        y2_win=y2_win*window
        
#        fft
        spec1=fft(y1_win,nfft) 
        spec1=spec1[0:nf]

        spec2=fft(y2_win,nfft) 
        spec2=spec2[0:nf]

        c11=np.abs(spec1)*np.abs(spec1)*2/(nfft*dt)   
        c22=np.abs(spec2)*np.abs(spec2)*2/(nfft*dt)    
        c12=spec1*np.conj(spec2)*2/(nfft*dt)    
#    
#    
#        stack power spec12um
        c11_stack=c11_stack+c11
        c22_stack=c22_stack+c22  
        c12_stack=c12_stack+c12
        
        itStart=itStart+nfft-nt_overlap
        itEnd=itStart+nfft
#        if the last segment, shift it to make sure to avoid padding
        if itEnd>ntTot-1:
            itEnd=ntTot-1
            itStart=itEnd-nfft
            isLast=True
        nwin_stack+=1
        
    c11_stack=c11_stack/nwin_stack
    c22_stack=c22_stack/nwin_stack
#
    c12_stack=c12_stack/nwin_stack
    
#    coh=np.abs(c12_stack)*np.abs(c12_stack)/(c11_stack*c22_stack);
#    L12 = c12_stack/c11_stack #(csr/crr)
#    c22_1=c22*(1-coh)
    
    L12,c22_1=multicoherence2_tf(c11_stack,c22_stack,c12_stack,vec_f,epsilon=1e-7)
    #   loop through for the second time to correct signal    
    itStart=0
    itEnd=nfft
#    nwin_stack=0
    isLast=False
    isEnd=False
    
    while not isEnd:
        if isLast:
            isEnd=True
#    while itEnd<=ntTot:

        y1_win=y1[itStart:itEnd];#%*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        y2_win=y2[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        if np.isinf(y1_win).any() or np.isinf(y2_win).any() or np.isnan(y1_win).any() or np.isnan(y2_win).any():
            continue
        
        if detrend:
            y1_win=signal.detrend(y1_win);
            y2_win=signal.detrend(y2_win);
        
#        window=blackman(nfft)
        
#        y1_win=y1_win*window
#        y2_win=y2_win*window
        
#        fft
        spec1=fft(y1_win,nfft) 
        spec1=spec1[0:nf]

        spec2=fft(y2_win,nfft) 
        spec2=spec2[0:nf]
        
#        spectrum of 2 after removing 1
        if type=='phase':
            spec2_1 = remove_coherence(spec1,spec2,L12,vec_f,fmin,fmax)
        elif type=='linear':
            spec2_1 = remove_coherence_linear(spec1,spec2,L12,vec_f,fmin,fmax)
        else: 
            print('invalid type')
#        c22_1=np.abs(spec2_1)**2
            
        y2_1[itStart:itEnd]=np.real(ifft(np.concatenate((spec2_1,np.conjugate(np.flipud(spec2_1[1:-1]))))))[0:len(y1_win)]
#        update indices
        itStart=itStart+nfft-nt_overlap
        itEnd=itStart+nfft
#        itEnd=itStart+nfft
        if itEnd>ntTot-1:   
            itEnd=ntTot-1
            itStart=itEnd-nfft
            isLast=True

        nwin_stack+=1

    
    return y2_1,c22_1



def remove_coh_signal4(x1,x2,x3,y,dt,ntwin,fmin,fmax,overlapPerc=0,detrend=True):
    """
    Remove part of signal y (response) that is coherent with x1,x2,x3 (source)
    Args:
        x1,x3,x3: signal
        y: targer response that is partially coherent with signal
    Returns:
        y_12,y_123,cyy_12,cyy_123
    """
    fs=1/dt
    if fmax>=fs/2:
        fmax=fs/2-0.01
        
#    determine total number of points and frequency vector
    ntTot=x2.shape[0]
    y_12=y.copy()
    y_123=y.copy()
    nfft=np.power(2,np.int(np.log2(ntwin))+1)
    nt_overlap= int(nfft*overlapPerc)
#    print(nfft)
#    print(nt_overlap)
    vec_f=np.linspace(0,1/2/dt,np.int(nfft//2)+1)
#    vec_f=fftfreq(nfft,dt)
    nf=np.int(nfft/2)+1;

#    initiate stacking
    c11_stack=np.zeros(nf)
    c22_stack=np.zeros(nf)
    c33_stack=np.zeros(nf)
    cyy_stack=np.zeros(nf)
    
#    cross-spectra
    c12_stack=np.zeros(nf)
    c13_stack=np.zeros(nf)
    c1y_stack=np.zeros(nf)
    c23_stack=np.zeros(nf)
    c2y_stack=np.zeros(nf)
    c3y_stack=np.zeros(nf)

    itStart=0
    itEnd=nfft
    nwin_stack=0
#   loop through for the first time    
#    while itEnd<=ntTot:        
    isLast=False
    isEnd=False
    while not isEnd:
        if isLast:
            isEnd=True     

        x1_win=x1[itStart:itEnd];#%*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        x2_win=x2[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        x3_win=x3[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        y_win=y[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);


        if (np.isinf(x1_win).any() or np.isinf(x2_win).any() or np.isinf(x3_win).any() or np.isinf(y_win).any() or 
            np.isnan(x1_win).any() or np.isnan(x2_win).any() or np.isnan(x3_win).any() or np.isnan(y_win).any()):
            continue
        x1_win=signal.detrend(x1_win)
        x2_win=signal.detrend(x2_win)
        x3_win=signal.detrend(x3_win)
        y_win=signal.detrend(y_win)
#        window=blackman(nfft)
#        window=hamming(nfft)  
        window=hamming(len(y_win))

#        window=hanning(nfft)     

        x1_win=x1_win*window
        x2_win=x2_win*window
        x3_win=x3_win*window
        y_win=y_win*window
        
#        fft
        spec1=fft(x1_win,nfft) 
        spec1=spec1[0:nf]

        spec2=fft(x2_win,nfft) 
        spec2=spec2[0:nf]

        spec3=fft(x3_win,nfft) 
        spec3=spec3[0:nf]
        
        spec_y=fft(y_win,nfft) 
        spec_y=spec_y[0:nf]
        
        
        c11=np.abs(spec1)*np.abs(spec1)*2/(nfft*dt)   
        c22=np.abs(spec2)*np.abs(spec2)*2/(nfft*dt)  
        c33=np.abs(spec3)*np.abs(spec3)*2/(nfft*dt)  
        cyy=np.abs(spec_y)*np.abs(spec_y)*2/(nfft*dt)  
        
        c12=spec1*np.conj(spec2)*2/(nfft*dt)    
        c13=spec1*np.conj(spec3)*2/(nfft*dt)    
        c1y=spec1*np.conj(spec_y)*2/(nfft*dt)   
        c23=spec2*np.conj(spec3)*2/(nfft*dt)   
        c2y=spec2*np.conj(spec_y)*2/(nfft*dt)
        c3y=spec3*np.conj(spec_y)*2/(nfft*dt)
        
#        stack power spec12um
        c11_stack=c11_stack+c11
        c22_stack=c22_stack+c22  
        c33_stack=c33_stack+c33
        cyy_stack=cyy_stack+cyy          
        
        c12_stack=c12_stack+c12
        c13_stack=c13_stack+c13
        c1y_stack=c1y_stack+c1y
        c23_stack=c23_stack+c23
        c2y_stack=c2y_stack+c2y
        c3y_stack=c3y_stack+c3y
        
        
        itStart=itStart+nfft-nt_overlap
        itEnd=itStart+nfft
#        itEnd=itStart+nfft
        if itEnd>ntTot-1:   
            itEnd=ntTot-1
            itStart=itEnd-nfft
            isLast=True

        nwin_stack+=1
        
#    end first loop
        
#   normalize
    c11_stack=c11_stack/nwin_stack
    c22_stack=c22_stack/nwin_stack
    c33_stack=c33_stack/nwin_stack
    cyy_stack=cyy_stack/nwin_stack

    c12_stack=c12_stack/nwin_stack
    c13_stack=c13_stack/nwin_stack
    c1y_stack=c1y_stack/nwin_stack
    c23_stack=c23_stack/nwin_stack
    c2y_stack=c2y_stack/nwin_stack
    c3y_stack=c3y_stack/nwin_stack

    
#    coh=np.abs(c12_stack)*np.abs(c12_stack)/(c11_stack*c22_stack);
#    L12 = c12_stack/c11_stack #(csr/crr)
#    c22_1=c22*(1-coh)
    
#    remove mutual coherence, calcualte transfer function
    l1y,l12,l13,l23_1,l2y_1,l3y_12=multicoherence4_tf(c11_stack,c22_stack,c33_stack,cyy_stack,c1y_stack,
                                                      c2y_stack,c3y_stack,c12_stack,c13_stack,c23_stack,vec_f)
    
    
    #   loop through for the second time to correct signal    
    itStart=0
    itEnd=nfft
#    nwin_stack=0
#    while itEnd<=ntTot:        
    isLast=False
    isEnd=False
    while not isEnd:
        if isLast:
            isEnd=True       

        x1_win=x1[itStart:itEnd];#%*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        x2_win=x2[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        x3_win=x3[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        y_win=y[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        
        if (np.isinf(x1_win).any() or np.isinf(x2_win).any() or np.isinf(x3_win).any() or np.isinf(y_win).any() or 
            np.isnan(x1_win).any() or np.isnan(x2_win).any() or np.isnan(x3_win).any() or np.isnan(y_win).any()):
            continue
        
        if detrend:
            x1_win=signal.detrend(x1_win)
            x2_win=signal.detrend(x2_win)
            x3_win=signal.detrend(x3_win)
            y_win=signal.detrend(y_win)
        
#        x1_win=x1_win*window
#        x2_win=x2_win*window
#        x3_win=x3_win*window
#        y_win=y_win*window
            
        
#        fft
        spec1=fft(x1_win,nfft) 
        spec1=spec1[0:nf]

        spec2=fft(x2_win,nfft) 
        spec2=spec2[0:nf]

        spec3=fft(x3_win,nfft) 
        spec3=spec3[0:nf]
        
        spec_y=fft(y_win,nfft) 
        spec_y=spec_y[0:nf]
        
#        spectrum of y after removing 12 or 123
#        spec2_1 = remove_coherence(spec1,spec2,L12,vec_f,fmin,fmax)
        spec_y_12,spec_y_123=remove_coherence4(spec1,spec2,spec3,spec_y,l1y,l12,l13,l23_1,l2y_1,l3y_12,vec_f,fmin,fmax)

        
        y_12[itStart:itEnd]=np.real(ifft(np.concatenate((spec_y_12,np.conjugate(np.flipud(spec_y_12[1:-1]))))))[0:len(y_win)]
        y_123[itStart:itEnd]=np.real(ifft(np.concatenate((spec_y_123,np.conjugate(np.flipud(spec_y_123[1:-1]))))))[0:len(y_win)]

        itStart=itStart+nfft-nt_overlap
        itEnd=itStart+nfft
#        itEnd=itStart+nfft
        if itEnd>ntTot-1:   
            itEnd=ntTot-1
            itStart=itEnd-nfft
            isLast=True

    
    return y_12,y_123


#    remove part of signal 2 (response) that is coherent with signal 1 (source)
def remove_coherence(spec_s,spec_r,Lrs,vec_f,fmin,fmax):
#    nf=len(vec_f)
#    nfft=(nf-1)*2
#    crr=np.abs(spec_r)*np.abs(spec_r)*2/(nfft*dt)   
#    css=np.abs(spec_s)*np.abs(spec_s)*2/(nfft*dt)    
#    crs=spec_r*np.conj(spec_s)*2/(nfft*dt) 
#    transfer function between 1 and 2
#    Lrs=crs/css
    win=(vec_f/fmin)**4/(1+(vec_f/fmin)**4)*(1/(1+(vec_f/fmax)**4))
    spec_r_s=spec_r-np.conj(Lrs*win)*spec_s
    return spec_r_s

#    remove part of signal 2 (response) that is coherent with signal 1 (source)
def remove_coherence_linear(spec_s,spec_r,Lrs,vec_f,fmin,fmax):
#    nf=len(vec_f)
#    nfft=(nf-1)*2
#    crr=np.abs(spec_r)*np.abs(spec_r)*2/(nfft*dt)   
#    css=np.abs(spec_s)*np.abs(spec_s)*2/(nfft*dt)    
#    crs=spec_r*np.conj(spec_s)*2/(nfft*dt) 
#    transfer function between 1 and 2
#    Lrs=crs/css
    win=(vec_f/fmin)**4/(1+(vec_f/fmin)**4)*(1/(1+(vec_f/fmax)**4))
    spec_r_s=spec_r-np.real(Lrs*win)*spec_s
    return spec_r_s


def multicoherence2_tf(g11,gyy,g1y,vf,epsilon=1e-7): 
    
#definition
#    c11=spec1*conj(spec1)
#    c12=spec1*conj(spec2)

#%these are the transfer functions between channels (1 and y);  (1 and 2);  (1 and 3).  
#    l12=g12/g11
    
#    minimum water level
    g11_wl=np.max((g11,epsilon*np.ones(g11.shape)),axis=0)
    gyy_wl=np.max((gyy,epsilon*np.ones(g11.shape)),axis=0)

    #%coherences between same
#    gam1y=abs(g1y)**2/(g11*gyy)
    l1y=g1y/g11_wl

    gam1y=abs(g1y)**2/(g11_wl*gyy_wl)

#    gam1y=abs(g1y)**2/(np.max(g11,eps)*gyy)
#    gam12=abs(g12)**2/(g11*g22)
    #%this is removing the effect of channel 1 from channels y, 2 and 3  
    gyy_1=gyy*(1-gam1y)

    return l1y,gyy_1

#from auto spectum and cospectra, calculate transfer function
# remove effect of 1,2 from y
def multicoherence3_tf(g11,g22,gyy,g1y,g2y,g12,vf): 
#definition
#    c11=spec1*conj(spec1)
#    c12=spec1*conj(spec2)

#%these are the transfer functions between channels (1 and y);  (1 and 2);  (1 and 3).  
    l1y=g1y/g11
    l12=g12/g11
    #%coherences between same
    gam1y=abs(g1y)**2/(g11*gyy)
    gam12=abs(g12)**2/(g11*g22)
    #%this is removing the effect of channel 1 from channels y, 2 and 3  
    gyy_1=gyy*(1-gam1y)
    g22_1=g22*(1-gam12)
    #%removing the effect of channel 1 from the cross spectra
    g2y_1=g2y-np.conj(l12)*g1y
    #%transfer function between 2 and 3 after removing the effect of channel 1
    l2y_1=g2y_1/g22_1
    #%coherence between (2 and y), (3 and y) and (2 and 3) after removing effect
    #%of channel 1
    gam2y_1=abs(g2y_1)**2/(g22_1*gyy_1)
     
    
    return l1y,l12,l2y_1,gam2y_1

def remove_coherence3(x1,x2,y,l1y,l12,l2y_1,vec_f,fmin,fmax):
    win=(vec_f/fmin)**4/(1+(vec_f/fmin)**4)*(1/(1+(vec_f/fmax)**4))
    
#    remove effect of x1 from y
    y_1=y-np.conj(l1y*win)*x1
#    remove effect of x1 from x2
    x2_1 = x2-np.conj(l12*win)*x1
#    remove effect of x1,x2 from y
    y_12=y_1-np.conj(l2y_1*win)*x2_1

    
    return y_12
   
    
#remove effects of 1,2,3 from y
def multicoherence4_tf_BACKUP(g11,g22,g33,gyy,g1y,g2y,g3y,g12,g13,g23,vf,epsilon=1e-7): 
#% In the notation below l3y_12 is the transfer function between 3 and y 
#    after removing coherent part of 1 and 2 from both the channels .

#definition
#    c11=spec1*conj(spec1)
#    c12=spec1*conj(spec2)
    
 
#%these are the transfer functions between channels (1 and y);  (1 and 2);  (1 and 3).  
    l1y=g1y/g11
    l12=g12/g11
    l13=g13/g11
    #%coherences between same
    gam1y=abs(g1y)**2/(g11*gyy)
    gam12=abs(g12)**2/(g11*g22)
    gam13=abs(g13)**2/(g11*g33)
    #%this is removing the effect of channel 1 from channels y, 2 and 3  
    gyy_1=gyy*(1-gam1y)
    g22_1=g22*(1-gam12)
    g33_1=g33*(1-gam13)
    #%removing the effect of channel 1 from the cross spectra
    g2y_1=g2y-np.conj(l12)*g1y
    g3y_1=g3y-np.conj(l13)*g1y
    g23_1=g23-np.conj(l12)*g13
    #%transfer function between 2 and 3 after removing the effect of channel 1
    l23_1=g23_1/g22_1
#    if(np.isinf(l23_1).any()):
#        pdb.set_trace()
    l2y_1=g2y_1/g22_1
    #%coherence between (2 and y), (3 and y) and (2 and 3) after removing effect
    #%of channel 1
    gam2y_1=abs(g2y_1)**2/(g22_1*gyy_1)
    gam3y_1=abs(g3y_1)**2/(g33_1*gyy_1)
    gam23_1=abs(g23_1)**2/(g33_1*g22_1)
     
    #%autospectra after removing effects of channels 1 and 2
    gyy_12=gyy_1*(1-gam2y_1)
    g33_12=g33_1*(1-gam23_1)
    g3y_12=g3y_1-np.conj(l23_1)*g2y_1
    #%coherence between 3 and y after removing effects of 2 and 3 
    gam3y_12=abs(g3y_12)**2/(g33_12*gyy_12)
    #%autospectra for y after removing effects of 1, 2 and 3 
    gyy_123=gyy_12*(1-gam3y_12)
    #%transfer function between 3 and y after removing the effects of channels 1
    #%and 2.  This is what is used in compliance where "y" is the vertical and 1
    #%and 2 are the two horizontals- the transfer function after removing the
    #%tilt noise... 
    l3y_12=g3y_12/g33_12
    
    return l1y,l12,l13,l23_1,l2y_1,l3y_12

#remove effects of 1,2,3 from y
def multicoherence4_tf(g11,g22,g33,gyy,g1y,g2y,g3y,g12,g13,g23,vf,epsilon=1e-7): 
#% In the notation below l3y_12 is the transfer function between 3 and y 
#    after removing coherent part of 1 and 2 from both the channels .

#definition
#    c11=spec1*conj(spec1)
#    c12=spec1*conj(spec2)
#    water lavel normalization 
    g11=np.max((g11,epsilon*np.ones(g11.shape)),axis=0)
    g22=np.max((g22,epsilon*np.ones(g22.shape)),axis=0)
    g33=np.max((g33,epsilon*np.ones(g33.shape)),axis=0)
    gyy=np.max((gyy,epsilon*np.ones(gyy.shape)),axis=0)

#%these are the transfer functions between channels (1 and y);  (1 and 2);  (1 and 3).  
    l1y=g1y/g11
    l12=g12/g11
    l13=g13/g11
    #%coherences between same
    gam1y=abs(g1y)**2/(g11*gyy)
    gam12=abs(g12)**2/(g11*g22)
    gam13=abs(g13)**2/(g11*g33)
    #%this is removing the effect of channel 1 from channels y, 2 and 3  
    gyy_1=gyy*(1-gam1y)
    g22_1=g22*(1-gam12)
    g33_1=g33*(1-gam13)
    #    water lavel normalization 
    gyy_1=np.max((gyy_1,epsilon*np.ones(gyy_1.shape)),axis=0)
    g22_1=np.max((g22_1,epsilon*np.ones(g22_1.shape)),axis=0)
    g33_1=np.max((g33_1,epsilon*np.ones(g33_1.shape)),axis=0)
    
    #%removing the effect of channel 1 from the cross spectra
    g2y_1=g2y-np.conj(l12)*g1y
    g3y_1=g3y-np.conj(l13)*g1y
    g23_1=g23-np.conj(l12)*g13
    #%transfer function between 2 and 3 after removing the effect of channel 1
    l23_1=g23_1/g22_1
    l2y_1=g2y_1/g22_1
    #%coherence between (2 and y), (3 and y) and (2 and 3) after removing effect
    #%of channel 1
    gam2y_1=abs(g2y_1)**2/(g22_1*gyy_1)
    gam3y_1=abs(g3y_1)**2/(g33_1*gyy_1)
    gam23_1=abs(g23_1)**2/(g33_1*g22_1)
     
    #%autospectra after removing effects of channels 1 and 2
    gyy_12=gyy_1*(1-gam2y_1)
    g33_12=g33_1*(1-gam23_1)
    
#    water level norm
    gyy_12=np.max((gyy_12,epsilon*np.ones(gyy_12.shape)),axis=0)
    g33_12=np.max((g33_12,epsilon*np.ones(g33_12.shape)),axis=0)
    
    g3y_12=g3y_1-np.conj(l23_1)*g2y_1
    #%coherence between 3 and y after removing effects of 2 and 3 
    gam3y_12=abs(g3y_12)**2/(g33_12*gyy_12)
    #%autospectra for y after removing effects of 1, 2 and 3 
    gyy_123=gyy_12*(1-gam3y_12)
    #%transfer function between 3 and y after removing the effects of channels 1
    #%and 2.  This is what is used in compliance where "y" is the vertical and 1
    #%and 2 are the two horizontals- the transfer function after removing the
    #%tilt noise... 
    l3y_12=g3y_12/g33_12
    
    return l1y,l12,l13,l23_1,l2y_1,l3y_12
def remove_coherence4(x1,x2,x3,y,l1y,l12,l13,l23_1,l2y_1,l3y_12,vec_f,fmin,fmax):
    win=(vec_f/fmin)**4/(1+(vec_f/fmin)**4)*(1/(1+(vec_f/fmax)**4))
    
#    remove effect of x1 from y
    y_1=y-np.conj(l1y*win)*x1
#    remove effect of x1 from x2
    x2_1 = x2-np.conj(l12*win)*x1
#    remove effect of x1,x2 from y
    y_12=y_1-np.conj(l2y_1*win)*x2_1

#    remove effect of 1,2, from 3
    x3_12=x3-np.conj(l13*win)*x1-np.conj(l23_1*win)*x2_1
    
    y_123=y_12-np.conj(l3y_12*win)*x3_12
    
    return y_12,y_123
   
#%% Spectral coherence

#def cepstrum(y):
#    spec1=fft()
def calc_coherence2(dt,y1,y2,ntwin,overlapPerc=0.0,epsilon=0.0000):
    ntTot=y1.size
#    vec_t=np.linspace(0,(nt-1)*dt,nt)

    #    next power of 2 
    nfft=np.power(2,np.int(np.log2(ntwin))+1)
    nt_overlap= int(nfft*overlapPerc)

    vec_f=np.linspace(0,1/2/dt,np.int(nfft//2)+1)
#    vec_f=fftfreq(nfft,dt)
    nf=np.int(nfft/2)+1;

    
    c11_stack=np.zeros(nf)
    c22_stack=np.zeros(nf)
    c12_stack=np.zeros(nf)
    nwin_stack =0

    itStart=0
    itEnd=nfft
    nwin_stack=0
    
    while itEnd<=ntTot:        

        y1_win=y1[itStart:itEnd];#%*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        y2_win=y2[itStart:itEnd];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        if np.isinf(y1_win).any() or np.isinf(y2_win).any() or np.isnan(y1_win).any() or np.isnan(y2_win).any():
            continue
        y1_win=signal.detrend(y1_win);
        y2_win=signal.detrend(y2_win);
#        window=blackman(nfft)
        window=hamming(nfft)

        y1_win=y1_win*window
        y2_win=y2_win*window
        
#        fft
        spec1=fft(y1_win,nfft) 
        spec1=spec1[0:nf]

        spec2=fft(y2_win,nfft) 
        spec2=spec2[0:nf]

        c11=np.abs(spec1)*np.abs(spec1)*2/(nfft*dt)   
        c22=np.abs(spec2)*np.abs(spec2)*2/(nfft*dt)    
        c12=spec1*np.conj(spec2)*2/(nfft*dt)    
#    
#    
#        stack power spec12um
        c11_stack=c11_stack+c11
        c22_stack=c22_stack+c22  
        c12_stack=c12_stack+c12
        
        itStart=itStart+nfft-nt_overlap
        itEnd=itStart+nfft
        nwin_stack+=1
        
    c11_stack=c11_stack/nwin_stack
    c22_stack=c22_stack/nwin_stack
#
    c12_stack=c12_stack/nwin_stack
    
    coh=np.abs(c12_stack)*np.abs(c12_stack)/(c11_stack*c22_stack);
    transferFunc = c12_stack/(c22_stack+c12_stack*epsilon);
    
    return vec_f,coh,transferFunc,nwin_stack
#  function to calculate spectral coherence from two time series
# inputs:
# vec_t time axis
# y1,y2: to time series to be compared
# nwin: number of windows
# ntwin: number of data points of each window
def calc_coherence(dt,y1,y2,nwin,ntwin,epsilon=0.0000):
    nt=y1.size
#    vec_t=np.linspace(0,(nt-1)*dt,nt)
    T_tot=nt*dt

    #    next power of
    nfft=np.power(2,np.int(np.log2(ntwin))+1)
    
    if nwin>1:
        t_overlap= np.ceil((nwin*nfft*dt-T_tot)/(nwin-1))
    else:
        t_overlap=0
#    print(t_overlap)
    # frequency axis
    vec_f=np.linspace(0,1/2/dt,np.int(nfft//2)+1)
#    vec_f=fftfreq(nfft,dt)
    nf=np.int(nfft/2)+1;

    
    c11_stack=np.zeros(nf)
    c22_stack=np.zeros(nf)
    c12_stack=np.zeros(nf)
    nwin_stack =0

    for iwin in range(1,nwin+1):
#        print(iwin)
        time_bg=(iwin-1)*(nfft*dt-t_overlap)

        
        idx_bg=np.int(time_bg/dt)
        idx_end=idx_bg+nfft

#        print(idx_bg)
#        print(idx_end)
#        select subset of data in moving window
#        ind_cut=(vec_t>=time_bg)&(vec_t<time_end)
    
        ind_cut=np.arange(idx_bg,idx_end)

        y1_win=y1[ind_cut];#%*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        y2_win=y2[ind_cut];# %*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        if np.isinf(y1_win).any() or np.isinf(y2_win).any() or np.isnan(y1_win).any() or np.isnan(y2_win).any():
            continue 
        y1_win=signal.detrend(y1_win);
        y2_win=signal.detrend(y2_win);
        window=hamming(nfft)
        
        y1_win=y1_win*window
        y2_win=y2_win*window
        
#        fft
        spec1=fft(y1_win,nfft) 
        spec1=spec1[0:nf]

        spec2=fft(y2_win,nfft) 
        spec2=spec2[0:nf]

        c11=np.abs(spec1)*np.abs(spec1)*2/(nfft*dt)   
        c22=np.abs(spec2)*np.abs(spec2)*2/(nfft*dt)    
        c12=spec1*np.conj(spec2)*2/(nfft*dt)    
#    
#    
#        stack power spec12um
        c11_stack=c11_stack+c11
        c22_stack=c22_stack+c22  
        c12_stack=c12_stack+c12
        
        nwin_stack=nwin_stack+1
        
    c11_stack=c11_stack/nwin_stack
    c22_stack=c22_stack/nwin_stack
#
    c12_stack=c12_stack/nwin_stack
    
    coh=np.abs(c12_stack)*np.abs(c12_stack)/(c11_stack*c22_stack);
    transferFunc = c12_stack/(c22_stack+c12_stack*epsilon);
    
    return vec_f,coh,transferFunc
    
    
#function to calculate average spectra
def calc_spec(dt,y1,nwin,ntwin,outputType='power'):
    nt=y1.size
    T_tot=nt*dt

    #    next power of
    nfft=np.power(2,np.int(np.log2(ntwin))+1)
    if nwin>1:
        t_overlap= np.ceil((nwin*nfft*dt-T_tot)/(nwin-1))
    else:
        t_overlap=0
        
    # frequency axis
    vec_f=np.linspace(0,1/2/dt,nfft//2+1)
    nf=nfft//2+1

#    nf    
    
    c11_stack=np.zeros(nf)
   
    nwin_stack =0

    for iwin in range(1,nwin+1):
        time_bg=(iwin-1)*(nfft*dt-t_overlap)
        
        idx_bg=np.int(time_bg/dt)
        idx_end=idx_bg+nfft
#        select subset of data in moving window
#        ind_cut=(vec_t>=time_bg)&(vec_t<time_end)
                    
        ind_cut=np.arange(idx_bg,idx_end)

        y1_win=y1[ind_cut];#%*flat_hanning(vec_t(ind_cut),0.2*nfft*dt);
        if np.isinf(y1_win).any() or np.isnan(y1_win).any():
            continue
        y1_win=signal.detrend(y1_win);
        window=hamming(nfft)
        
        y1_win=y1_win*window
#        y2_win=y2_win*window
        
#        fft
        spec1=fft(y1_win,nfft) 
        spec1=spec1[0:nf]


        c11=np.abs(spec1)*np.abs(spec1)*2/(nfft*dt)   
#    
#    
#        stack power spec12um
        c11_stack=c11_stack+c11
        
        nwin_stack=nwin_stack+1
        
    if nwin_stack==0:
#        pdb.set_trace()   
        c11_stack=c11_stack*0
    else:
        c11_stack=c11_stack/nwin_stack    

    if outputType=='power':
        y=c11_stack
    else:
        y=np.sqrt(c11_stack)    
        
    return vec_f,y  
