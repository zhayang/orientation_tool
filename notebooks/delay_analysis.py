#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:25:15 2018

@author: zhay
"""
# a script to calculate delay time between axial and torsional wave
#%% load required libraries
#import imp
import os
import scipy as sp
#import pybrain as pb
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from spectral import calc_coherence,calc_spec,calc_coherence2,remove_coh_signal2,remove_coh_signal4
from scipy.signal import butter, lfilter, filtfilt
from signal_util import calcCCFStack,calcCCF,getRicker
from scipy.fftpack import fft,ifft,fftfreq,fftshift

# %matplotlib inline
#%matplotlib notebook
#import os
import pickle
#import h5py
#import numpy as np
from time import time
import DeepDataFrame
import pdb
#%autoreload 2
import deep_util




#%% load data

#datafile='/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/BITTERLY_OWENS_COLIN_A_ULW_1_merge_cut_100hz_learning.csv'
datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_30/RUCKMAN_RANCH_30_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_32/RUCKMAN_RANCH_32_merge_cut_100hz_learning.csv'
#datafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_35/RUCKMAN_RANCH_35_merge_cut_100hz_learning.csv'

data=deep_util.loadDEEPData(datafile,savePickle=True)



if('01_Global-Generic_Surface-HOLEDEPTH' in data.columns):
    device='01_Global-Generic_Surface'
    holedepthName=device+'-HOLEDEPTH'
    bitdepthName=device+'-BIT_DEPTH'
else:
    device='01_GLOBAL_GENERIC_SURFACE'
    holedepthName=device+'-HOLEDEPTH'
    bitdepthName=device+'-BIT_DEPTH'

#wobName='01_Global-Generic_Surface-SWOB'
#rigstateName='01_Global-Generic_Surface-copRigState'


deeptime=(data.index).to_series()




#load EMS 1hz data
##bitter owen colin A
#emsdatafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/EMS/BITTERLY_OWENS_COLIN_A_ULW#1_BHA02R01_EMS01_SNEMSIB38_MEMORYMERGED.csv'
#emshifidatafile = '/home/zhay/DEEP/deep_data/BITTERLY_OWENS_COLIN_A_ULW_1/EMS/bha02_merge_hifi.csv'
#timeshiftHour=0
#timeshiftSecond=204
#
#
#RUCKMAN_RANCH_30
emsdatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_30/ems/merge_ems_1hz.csv'
emshifidatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_30/ems/merge_ems_hifi.csv'
timeshiftHour=5
timeshiftSecond=56

#RUCKMAN_RANCH_35
#emsdatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_35/ems/merge_ems_1hz.csv'
#emshifidatafile='/home/zhay/DEEP/deep_data/RUCKMAN_RANCH_35/ems/merge_ems_hifi.csv'
#timeshiftHour=5
#timeshiftSecond=65


emsdata=deep_util.loadDEEPData(emsdatafile,timeColumnName='TIME(datetime)',
                               timeshiftHour=timeshiftHour, timeshiftSecond=timeshiftSecond)

emsdatahifi=deep_util.loadDEEPData(emshifidatafile,timeColumnName='TIME(datetime)',
                               timeshiftHour=timeshiftHour, timeshiftSecond=timeshiftSecond)
#emsPickleFileName=emsdatafile+'.p'
#ReadFromCsv=True
#if os.path.isfile(emsPickleFileName)<0:
#    print('pickle file does not exist, load from csv')
#    emsdataraw=pd.read_csv(emsdatafile,skiprows=[1])
#
#
#    timestamp_ems = pd.to_datetime(emsdataraw['TIME(datetime)'])
#    timestamp_ems=timestamp_ems-pd.offsets.Second(timeshiftSecond)-pd.offsets.Hour(timeshiftHour)
#    emsdataraw['TIME(datetime)']=timestamp_ems
#    emsdata=emsdataraw.set_index('TIME(datetime)')    
#    
#    with open(emsPickleFileName,'wb') as file:
#        pickle.dump(emsdata,file)
#else:
#    t0 = time()
#    with open(emsPickleFileName,'rb') as file:
#        emsdata=pickle.load(file)
#    print('done reading pickle in %.2fs.' % (time() - t0))
#    
#if  os.path.isfile(emshifidatafile+'.p')==False:

    
#if read hifi files
#    emsdataraw=pd.read_csv(emshifidatafile,skiprows=[1])
#
#
#    timestamp_ems = pd.to_datetime(emsdataraw['TIME(datetime)'])
#    timestamp_ems=timestamp_ems-pd.offsets.Second(timeshiftSecond)-pd.offsets.Hour(timeshiftHour)
#    emsdataraw['TIME(datetime)']=timestamp_ems
#    emsdatahifi=emsdataraw.set_index('TIME(datetime)')    
#    
#    with open(emshifidatafile+'.p','wb') as file:
#        pickle.dump(emsdatahifi,file)
#else:
#    t0 = time()
#    with open(emshifidatafile+'.p','rb') as file:
#        emsdatahifi=pickle.load(file)
#    print('done reading pickle in %.2fs.' % (time() - t0))
    
    

emsLatAccelXRaw=emsdata['EMS_LATX_MAX(G)']
emsLatAccelX = emsLatAccelXRaw[~np.isnan(emsLatAccelXRaw)]

emsLatAccelYRaw=emsdata['EMS_LATY_MAX(G)']
emsLatAccelY = emsLatAccelYRaw[~np.isnan(emsLatAccelYRaw)]

emsLatAccelTRaw=emsdata['EMS_LLAT_MAX(G)']
emsLatAccelT = emsLatAccelTRaw[~np.isnan(emsLatAccelTRaw)]

emsLatAccelT_aligned=emsLatAccelT.reindex(deeptime).interpolate()

ems_gx= emsdatahifi['GX(G)']
ems_gy= emsdatahifi['GY(G)']
ems_gz= emsdatahifi['AXIAL_VIBRATION(G)']

ems_gx_100hz=ems_gx.resample('10ms').mean().interpolate(method='linear')
ems_gx_aligned=ems_gx_100hz.reindex(deeptime).interpolate()

ems_gy_100hz=ems_gy.resample('10ms').mean().interpolate(method='linear')
ems_gy_aligned=ems_gy_100hz.reindex(deeptime).interpolate()

ems_gz_100hz=ems_gz.resample('10ms').mean().interpolate(method='linear')
ems_gz_aligned=ems_gz_100hz.reindex(deeptime).interpolate()

#%% set depth range and channels
depthMin=5000/3.28;
depthMax=10000/3.28;
mask= (data[holedepthName]>depthMin) & (data[holedepthName]< depthMax)
#mask= (data['01_Global.Pason_Surface.holeDepth']>depthMin) & (data['01_Global.Pason_Surface.holeDepth']< depthMax)
datacut = data.loc[mask]
deeptime_cut=(datacut.index).to_series()

#assign channels
torque=datacut['01_GLOBAL_PASON_TTS-STRQ']
tension=datacut['01_GLOBAL_PASON_TTS-TENSION']
pressure=datacut['01_GLOBAL_PASON_TTS-PRESSURE']
drpm=datacut['01_GLOBAL_PASON_TTSV-DRPM']


holedepth = datacut[device+'-HOLEDEPTH']
rigstate=datacut[device+'-copRigState']
bitdepth=datacut[device+'-BIT_DEPTH']
srpm=datacut[device+'-SRPM']

ar_tts=datacut['01_GLOBAL_PASON_TTS-RAD_ACCELERATION']
at_tts=datacut['01_GLOBAL_PASON_TTS-TAN_ACCELERATION']
az_tts=datacut['01_GLOBAL_PASON_TTS-AXIAL_ACCELERATION']



latVibration=emsLatAccelT_aligned.loc[mask]
ems_gz=ems_gz_aligned.loc[mask]*9.8
ems_gx=ems_gx_aligned.loc[mask]*9.8
ems_gy=ems_gy_aligned.loc[mask]*9.8


#%% loop through each window to calculate cohereance cross-correlation spectra, stacking within window


fs=100
dt=1/fs
# number of samples within each long window (need to be short so that relationship is constant)
ntwin=100*fs
# number of samples within each short windows (long enough to cover possible delays)
ntSegment=10*fs 

#step size, can be shorter than ntwin to allow overalpping windows
ntStep=100*fs

ntTotal=datacut.shape[0]
nwin=(ntTotal-ntwin)//ntStep

mtx_ccf=np.zeros((nwin,ntSegment*2-1))
mtx_ccf_r=np.zeros((nwin,ntSegment*2-1))

vec_rigstate=np.zeros((nwin,))
depthArray=np.zeros((nwin,))
bitDepthArray=np.zeros((nwin,))
wobArray=np.zeros((nwin,))
srpmArray=np.zeros((nwin,))
maxAccelArray=np.zeros((nwin,))
meanAccelArray=np.zeros((nwin,))
medianAccelArray=np.zeros((nwin,))

cohfreqList=[]
cohList=[]
cohList_r=[]
transList=[]
transList_r=[]
spec1List=[]
spec2List=[]
spec1List_r=[]
spec2List_r=[]
vec_t=np.linspace(0,ntwin*dt,ntwin)

for iwin in range(0,nwin):
    itBegin=iwin*ntStep
    itEnd = itBegin + ntwin
    srpm_win= np.array(srpm[itBegin:itEnd])
    torque_win = np.array(torque[itBegin:itEnd])
    tension_win=np.array(tension[itBegin:itEnd])
    rigstate_win=np.array(rigstate[itBegin:itEnd])

    pressure_win=np.array(pressure[itBegin:itEnd])
    ar_tts_win=np.array(ar_tts[itBegin:itEnd])
    at_tts_win=np.array(at_tts[itBegin:itEnd])
    az_tts_win=np.array(az_tts[itBegin:itEnd])      
    ah_tts_win=np.sqrt(at_tts_win**2+ar_tts_win**2)
    
    latVibration_win=np.array(latVibration[itBegin:itEnd])
    
    ax_ems_win=np.array(ems_gx[itBegin:itEnd])
    ay_ems_win=np.array(ems_gy[itBegin:itEnd])
    az_ems_win=np.array(ems_gz[itBegin:itEnd]) 
    ah_ems_win=np.sqrt(ay_ems_win**2+ax_ems_win**2)

    vec_rigstate[iwin]=sp.stats.mode(rigstate_win)[0][0]
    
    depthArray[iwin]=np.nanmedian(holedepth[itBegin:itEnd])
    bitDepthArray==np.nanmedian(bitdepth[itBegin:itEnd])
    srpmArray[iwin]=np.nanmedian(srpm_win)
    maxAccelArray[iwin]=np.nanmax(latVibration_win)
    meanAccelArray[iwin]=np.nanmean(latVibration_win)    
    medianAccelArray[iwin]=np.nanmedian(latVibration_win)

    
#    calculate cross-correlation between axial and tangential
    
#    only calculate for rotary drilling
    if vec_rigstate[iwin]>=0 and np.std(tension_win)>1e-5: 
#        ccf,lag=calcCCF(az_tts_win,at_tts_win,dt=0.01,fmin=10,fmax=40)
#        y1=az_tts_win
        
        y1=torque_win
        
#        y2=az_ems_win
#        y2=np.sqrt(at_tts_win**2+ar_tts_win**2)
        y2=tension_win
        
#        y2=np.sqrt(ax_ems_win**2+ay_ems_win**2)
        f1Plot=5
        f2Plot=40
#        y1=getRicker(1,vec_t,t0=1) + np.random.rand(len(y1))*0.001
#        y2=getRicker(1,vec_t,t0=3)+np.random.rand(len(y1))*0.001
#        ccf,lag=calcCCF(y1,y2,dt=0.01,fmin=0.5,fmax=5)
#        vec_f,coh,trans,nstack=calc_coherence2(dt,y1,y2,ntSegment)
        
       

        ccf,lag=calcCCFStack(y1,y2,ntSegment,dt=0.01,fmin=f1Plot,fmax=f2Plot,overlapPerc=0)        
#        ccf,lag=calcCCFStack(y1,y2,ntSegment,dt=0.01,fmin=f1,fmax=f2,overlapPerc=0,normalize=False)        


#        y1_r,spec1_r=remove_coh_signal2(pressure_win,y1,dt,ntSegment,
#                            fminRemove,fmaxRemove,overlapPerc=0.0,detrend=False)
#        y1_r,spec1_r=remove_coh_signal2(torque_win,y1,dt,ntSegment,fmin=1,fmax=40,overlapPerc=0.0,detrend=False,type='linear')
        y1_r,spec1_r=remove_coh_signal2(pressure_win,y1,dt,ntSegment,fmin=1,fmax=40,overlapPerc=0.0,detrend=False)
        y2_r,spec2_r=remove_coh_signal2(pressure_win,y2,dt,ntSegment,fmin=1,fmax=40,overlapPerc=0.0,detrend=False)
#        if(np.std(pressure_win)==0):
#            print('flat line')
#            pdb.set_trace()
            
#        _,y1_r = remove_coh_signal4(pressure_win,at_tts_win,ar_tts_win,y1,dt,ntSegment,fmin=1,fmax=40,overlapPerc=0.0,detrend=False)
#        _,y2_r = remove_coh_signal4(pressure_win,at_tts_win,ar_tts_win,y2,dt,ntSegment,fmin=1,fmax=40,overlapPerc=0.0,detrend=False)
        
        ccf_r,lag=calcCCFStack(y1_r,y2_r,ntSegment,dt=0.01,fmin=f1Plot,fmax=f2Plot,overlapPerc=0) 
        
        vec_f,spec1=calc_spec(dt,y1,ntwin//ntSegment,ntSegment,outputType='power')
        _,spec2=calc_spec(dt,y2,ntwin//ntSegment,ntSegment,outputType='power')
        
        if np.isinf(y1_r).any() or np.isinf(y2_r).any() or np.isinf(y1_r).any() or np.isinf(y2_r).any(): 
            transList.append(vec_f*0)
            transList_r.append(vec_f*0)
    
            cohList.append(vec_f*0)
            cohList_r.append(vec_f*0)
    
            spec1List.append(vec_f*0)
            spec2List.append(vec_f*0)
            spec1List_r.append(vec_f*0)
            spec2List_r.append(vec_f*0)
            continue
        


        _,spec1_r=calc_spec(dt,y1_r,ntwin//ntSegment,ntSegment,outputType='power')
        _,spec2_r=calc_spec(dt,y2_r,ntwin//ntSegment,ntSegment,outputType='power')
        
        vec_f,coh,trans=calc_coherence(dt,y1,y2,ntwin//ntSegment,ntSegment)
        vec_f,coh_r,trans_r,nstack=calc_coherence2(dt,y1_r,y2_r,ntSegment)
 
        cohfreqList.append(vec_f)
        cohList.append(coh)
        cohList_r.append(coh_r)
        transList.append(trans)
        transList_r.append(trans_r)
        mtx_ccf[iwin,:] = ccf
        mtx_ccf_r[iwin,:] = ccf_r

        spec1List.append(spec1)
        spec2List.append(spec2)

        spec1List_r.append(spec1_r)
        spec2List_r.append(spec2_r)
        
        if(iwin%1000==0):
            print('processing window %3.0f out of %5.0f'%(iwin,nwin))
#        if(iwin%2000==0):
#            plt.figure()     
#            ax1=plt.subplot(311)
#            ax1.plot(y2/np.max(np.abs(y2)),label=' normalized y2')
#            ax1.plot(y1/np.max(np.abs(y1)),label=' normalized y1.')
#
#            plt.xlabel('time')
#            plt.title('Depth is %5.2f ft'%(3.28*depthArray[iwin]))
#            ax1.legend()
#            ax1b=ax1.twinx()
#            ax1b.plot(srpm_win,'-r',label='RPM')
#            plt.ylim(0,100)
#            ax1b.legend()
#            ax2=plt.subplot(312)
#            ax2.plot(lag,ccf,label='ccf')
#            plt.xlim(-5,5)
#            plt.xlabel('lab time (sec)')
#            ax3=plt.subplot(313)
#            ax3.semilogx(vec_f,coh,'-r',label='coherence')
##            ax3.plot(vec_f,coh,'-r',label='coherence')
#            plt.legend()
#            ax3b=ax3.twinx()
#            ax3b.loglog(vec_f,spec1,label='spectrum of y1')
#            ax3b.loglog(vec_f,spec2,label='spectrum of y2')
#            
##            ax3b.semilogx(vec_f,np.angle(trans),label='phase')
#
#            plt.legend()
#            plt.xlabel('frequency')
        if(iwin%1000==0):
            b, a = signal.butter(4, np.array([f1Plot,f2Plot])/(fs/2), 'bandpass')
            
            y1_filt=filtfilt(b, a, y1)
            y1_r_filt=filtfilt(b, a, y1_r)
            y2_filt=filtfilt(b, a, y2)
            
            
            plt.figure()     
            ax1=plt.subplot(311)
            ax1.plot(y1_filt/np.max(np.abs(y1_filt)),'-g',label='y1')
            ax1.plot(2+y2_filt/np.max(np.abs(y2_filt)),label='y2')
            ax1.plot(y1_r_filt/np.max(np.abs(y1_filt)),label='y1-noise')
            plt.xlabel('time')
            ax1.legend()
            ax1b=ax1.twinx()
            #ax1b.plot(torque,label='torque')
            ax1b.legend()
            plt.title('Depth is %5.2f ft'%(3.28*depthArray[iwin]))

            
            ax2=plt.subplot(312)
            ax2.plot(lag,ccf,label='ccf')
            ax2.plot(lag,ccf_r,label='ccf_processed')
            ax2.legend()
            
            plt.xlim(-5,5)
            plt.xlabel('lab time (sec)')
            ax3=plt.subplot(313)
            ax3.semilogx(vec_f,coh,'-r',label='coherence')
            #            ax3.plot(vec_f,coh,'-r',label='coherence')
            plt.legend()
            ax3b=ax3.twinx()
            ax3b.loglog(vec_f,spec1,label='spectrum of y1')
            ax3b.loglog(vec_f,spec2,label='spectrum of y2')
            #            ax3b.loglog(vec_f,spec1_r,label='spectrum of y1 after removing noise')
            ax3b.loglog(vec_f,spec1_r,label='spectrum of y1 after removing noise')
            #            ax3b.semilogx(vec_f,np.angle(trans),label='phase')
            
            ax3b.legend()
    else:
        transList.append(vec_f*0)
        transList_r.append(vec_f*0)

        cohList.append(vec_f*0)
        cohList_r.append(vec_f*0)

        spec1List.append(vec_f*0)
        spec2List.append(vec_f*0)
        spec1List_r.append(vec_f*0)
        spec2List_r.append(vec_f*0)

#        transList.append(trans)

    

    #%% plotting
vp=16800
vs=10400


    #------------organize transfer function
mtx_resp=np.zeros((nwin,(len(vec_f)-1)*2),dtype=complex)
mtx_transfer=np.zeros((nwin,len(vec_f)),dtype=complex)
mtx_coh=np.zeros((nwin,len(vec_f)))
mtx_spec1=np.zeros((nwin,len(vec_f)),dtype=complex)
mtx_spec2=np.zeros((nwin,len(vec_f)),dtype=complex)

freqwin=(vec_f/f1Plot)**4/(1+(vec_f/f1Plot)**4)*(1/(1+(vec_f/f2Plot)**4))

for iwin in range(nwin):
#    filter transer function in desired band
    trans_window=transList_r[iwin]*freqwin 
#    trans_window=transList[iwin]*freqwin 

#    trans_ifft=np.concatenate((transList[iwin],np.conjugate(np.flipud(transList[iwin][1:-1]))))
    trans_ifft=np.concatenate((trans_window,np.conjugate(np.flipud(trans_window[1:-1]))))

    mtx_resp[iwin,:]=fftshift(ifft(trans_ifft))
    mtx_transfer[iwin,:]=transList[iwin]
    mtx_coh[iwin,:]=cohList_r[iwin]
    mtx_spec1[iwin,:]=spec1List_r[iwin]
    mtx_spec2[iwin,:]=spec2List_r[iwin]
#------------organize transfer function

    
#zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
#depthInt=np.arange(depthArray[0], depthArray[-1],10)
#lagInt=lag
#dataInt=griddata((depthArray,lag),mtx_ccf.transpose(),(depthInt,lagInt),method='cubic')

delta_slowness=(1/vs-1/vp)
depth_extents = depthArray[0]*3.281, depthArray[-1]*3.281, lag[0]/delta_slowness, lag[-1]/delta_slowness
#depth_extents = depthArray[0]*3.281, depthArray[-1]*3.281, lag[0]*vp, lag[-1]*vp
        
extents = depthArray[0]*3.281, depthArray[-1]*3.281, lag[0], lag[-1]
freq_extents = depthArray[0]*3.281, depthArray[-1]*3.281, vec_f[0], vec_f[-1]



fig=plt.figure()        
fig.set_size_inches(30, 30, forward=True)
ax1=plt.subplot(221)
#plt.imshow(np.flipud(np.real(mtx_resp).transpose()),as'pect='auto')
#plt.imshow(mtx_ccf.transpose(),origin='lower',aspect='auto',extent=extents);plt.ylim(-2,2);plt.clim(-0.1,0.1)
X,Y=np.meshgrid(depthArray*3.281,lag)
xp=np.append(depthArray*3.281,depthArray[-1]*3.281)

# ------Option A plot cross-correlation
yp=np.append(lag,lag[-1])
mesh1=ax1.pcolorfast(xp,yp,mtx_ccf.transpose(),vmin=-0.1,vmax=0.1);mesh1.set_cmap('gray');plt.ylim(-2,2);plt.suptitle('raw')
#mesh1=ax1.pcolorfast(xp,yp,mtx_ccf_r.transpose(),vmin=-0.1,vmax=0.1);mesh1.set_cmap('gray');plt.ylim(-1,1);plt.suptitle('remove pressure noise')

#mtx_ccf=mtx_ccf/np.mean(np.abs(mtx_ccf))

#-----------Option B plot ifft of transfer function
#yp=np.arange(-mtx_resp.shape[1]//2,mtx_resp.shape[1]//2+1)*dt
#mesh1=ax1.pcolorfast(xp,yp,np.real(mtx_resp).transpose(),vmin=-0.5,vmax=.5);plt.ylim(-1,1);mesh1.set_cmap('gray');plt.suptitle('remove noise')
#mesh1=ax1.pcolorfast(xp,yp,np.real(mtx_resp).transpose(),vmin=-0.5,vmax=0.5);plt.ylim(-1,1);mesh1.set_cmap('gray');plt.suptitle('raw')

#plt.clim(-1,1)
#ax1.plot(xp,-xp/vs,'--r')  
#ax1.plot(xp,-2*(xp)/vs,'--r')   
#ax1.plot(xp,-4*(xp)/vs,'--r')   

#ax1.plot(xp,xp/vp,'--r')   

#ax1.plot(xp,-xp*3/vp,'--r')        
#ax1.plot(xp,-xp*5/vp,'--r')   
ax1.plot(xp,-xp*delta_slowness,'--r')   
#ax1.plot(xp,-11500*np.ones(xp.shape)*delta_slowness,'--b')   

#ax1.plot(depthArray*3.281,maxAccelArray,'-ro')   
#ax1.plot(depthArray*3.281,medianAccelArray,'-ro')   

#ax1.plot(xp,-xp*3*delta_slowness,'--r')   
#ax1.plot(xp,-xp*5*delta_slowness,'--r')   
#ax1.plot(xp,-xp*7*delta_slowness,'--r')   

#plt.ylim(-20000,20000)



plt.ylabel('Time lag',fontsize=20)      
plt.xlabel('Depth',fontsize=20)
plt.title('Cross-correlation between %2.2f and %2.2f Hz'%(f1Plot,f2Plot),fontsize=20)
#plt.imshow(np.flipud(np.real(mtx_resp).transpose()),aspect='auto')

#plt.imshow(np.flipud(mtx_ccf.transpose()),aspect='auto')
#plt.imshow(np.flipud(mtx_ccf.transpose()),aspect='auto')
ax2=plt.subplot(222,sharex=ax1)
xp=np.append(depthArray*3.281,depthArray[-1]*3.281)
yp=np.append(vec_f,vec_f[-1])

#plt.imshow(np.flipud(np.abs(mtx_transfer).transpose()),aspect='auto',extent=freq_extents);plt.clim(0,20)
#c2=ax2.pcolorfast(xp,yp,np.abs(mtx_transfer).transpose(),vmin=0,vmax=20);
mesh2=ax2.pcolorfast(xp,yp,np.abs(mtx_coh).transpose(),vmin=0,vmax=0.6);plt.ylim(0,50)
#fig.colorbar(mesh2)


plt.ylabel('Freqnecy',fontsize=20)
plt.xlabel('Depth',fontsize=20)
plt.title('Coherence',fontsize=20)



# plot spectum with depth
#fig=plt.figure()        
#fig.set_size_inches(30, 15, forward=True)
ax3=plt.subplot(223,sharex=ax1,sharey=ax2)
xp=np.append(depthArray*3.281,depthArray[-1]*3.281)     
yp=np.append(vec_f,vec_f[-1])

mesh3=ax3.pcolorfast(xp,yp,np.log10(np.abs(mtx_spec1)).transpose(),vmin=6,vmax=10);#plt.ylim(-5,5)
plt.ylabel('frequency',fontsize=20)
plt.xlabel('Depth',fontsize=20)
plt.title('Channel 1 spectrum',fontsize=20)
#plt.imshow(np.flipud(np.real(mtx_resp).transpose()),aspect='auto')

#plt.imshow(np.flipud(mtx_ccf.transpose()),aspect='auto')
#plt.imshow(np.flipud(mtx_ccf.transpose()),aspect='auto')
ax4=plt.subplot(224,sharex=ax1,sharey=ax2)

mesh4=ax4.pcolorfast(xp,yp,np.log10(np.abs(mtx_spec2)).transpose(),vmin=-2,vmax=5);#plt.ylim(-5,5)
#fig.colorbar(c2)
plt.ylabel('Freqnecy',fontsize=20)
plt.xlabel('Depth',fontsize=20)
plt.title('Channel 2 spectrum',fontsize=20)

#%% loop through each window to remove coherent signal in ax,at from tension and torque


fs=100
dt=1/fs
# number of samples within each long window (need to be short so that relationship is constant)
ntwin=100*fs
# number of samples within each short windows (long enough to cover possible delays)
ntSegment=10*fs 

#step size, can be shorter than ntwin to allow overalpping windows
ntStep=100*fs

ntTotal=datacut.shape[0]
nwin=(ntTotal-ntwin)//ntStep

mtx_ccf=np.zeros((nwin,ntSegment*2-1))
mtx_ccf_r=np.zeros((nwin,ntSegment*2-1))

vec_rigstate=np.zeros((nwin,))
depthArray=np.zeros((nwin,))
bitDepthArray=np.zeros((nwin,))
wobArray=np.zeros((nwin,))
srpmArray=np.zeros((nwin,))
maxAccelArray=np.zeros((nwin,))
meanAccelArray=np.zeros((nwin,))
medianAccelArray=np.zeros((nwin,))

cohfreqList=[]
cohList=[]
transList=[]
transList_r=[]
spec1List=[]
spec2List=[]
vec_t=np.linspace(0,ntwin*dt,ntwin)

for iwin in range(0,nwin):
    itBegin=iwin*ntStep
    itEnd = itBegin + ntwin
    srpm_win= np.array(srpm[itBegin:itEnd])
    torque_win = np.array(torque[itBegin:itEnd])
    tension_win=np.array(tension[itBegin:itEnd])
    rigstate_win=np.array(rigstate[itBegin:itEnd])

    
    ar_tts_win=np.array(ar_tts[itBegin:itEnd])
    at_tts_win=np.array(at_tts[itBegin:itEnd])
    az_tts_win=np.array(az_tts[itBegin:itEnd])   
    
    ax_ems_win=np.array(ems_gx_aligned[itBegin:itEnd])
    ay_ems_win=np.array(ems_gy_aligned[itBegin:itEnd])
    az_ems_win=np.array(ems_gz_aligned[itBegin:itEnd]) 
    
    latVibration_win=np.array(latVibration[itBegin:itEnd])
    pressure_win=np.array(pressure[itBegin:itEnd])
    

    vec_rigstate[iwin]=sp.stats.mode(rigstate_win)[0][0]
    
    depthArray[iwin]=np.nanmedian(holedepth[itBegin:itEnd])
    bitDepthArray==np.nanmedian(bitdepth[itBegin:itEnd])
    srpmArray[iwin]=np.nanmedian(srpm_win)
    maxAccelArray[iwin]=np.nanmax(latVibration_win)
    meanAccelArray[iwin]=np.nanmean(latVibration_win)    
    medianAccelArray[iwin]=np.nanmedian(latVibration_win)

    
#    calculate cross-correlation between axial and tangential
    
#    only calculate for rotary drilling
    if vec_rigstate[iwin]>=0: 
#        ccf,lag=calcCCF(az_tts_win,at_tts_win,dt=0.01,fmin=10,fmax=40)
#        y1=az_tts_win
#        y2=at_tts_win
        
        y1=tension_win
#        y2=np.sqrt(at_tts_win**2+ar_tts_win**2)
        y2=torque_win
        
        f1Plot=4
        f2Plot=40
#        y1=getRicker(1,vec_t,t0=1) + np.random.rand(len(y1))*0.001
#        y2=getRicker(1,vec_t,t0=3)+np.random.rand(len(y1))*0.001
#        ccf,lag=calcCCF(y1,y2,dt=0.01,fmin=0.5,fmax=5)
#        vec_f,coh,trans=calc_coherence(dt,y1,y2,ntwin//ntSegment,ntSegment)

        fminRemove=2
        fmaxRemove=40

#        pdb.set_trace()        
#        y2_12,y2_123=remove_coh_signal4(at_tts_win,ar_tts_win,az_tts_win,tension_win,dt,ntSegment,
#                                        fmin,fmax,overlapPerc=0.0,detrend=False)
        
#        y1_12,y1_r=remove_coh_signal4(torque_win,pressure_win,tension_win,y1,dt,ntSegment,
#                            fminRemove,fmaxRemove,overlapPerc=0.0,detrend=False)
        y1_r,spec1_r=remove_coh_signal2(pressure_win,y1,dt,ntSegment,
                            fminRemove,fmaxRemove,overlapPerc=0.0,detrend=False)
#        y1_r,spec1_r=remove_coh_signal2(tension_win,y1,dt,ntSegment,fmin,fmax,overlapPerc=0.0,detrend=False)
        
        ccf_r,lag=calcCCFStack(y1_r,y2,ntSegment,dt=0.01,fmin=f1Plot,fmax=f2Plot,overlapPerc=0) 
        ccf,lag=calcCCFStack(y1,y2,ntSegment,dt=0.01,fmin=f1Plot,fmax=f2Plot,overlapPerc=0)        

#        ccf,lag=calcCCFStack(y1,y2,ntSegment,dt=0.01,fmin=f1,fmax=f2,overlapPerc=0,normalize=False)        

        _,spec1=calc_spec(dt,y1,ntwin//ntSegment,ntSegment,outputType='power')
        _,spec2=calc_spec(dt,y2,ntwin//ntSegment,ntSegment,outputType='power')
        _,spec1_r=calc_spec(dt,y1_r,ntwin//ntSegment,ntSegment,outputType='power')
        
        vec_f,coh,trans,nstack=calc_coherence2(dt,y1,y2,ntSegment)
        vec_f,coh,trans_r,nstack=calc_coherence2(dt,y1_r,y2,ntSegment)
        
        cohfreqList.append(vec_f)
        cohList.append(coh)
        transList.append(trans)
        transList_r.append(trans_r)

        mtx_ccf[iwin,:] = ccf
        mtx_ccf_r[iwin,:]=ccf_r
        spec1List.append(spec1)
        spec2List.append(spec2)
        
        if(iwin%100==0):
            print('processing window %3.0f out of %5.0f'%(iwin,nwin))
            
        if(iwin%500==0):
            b, a = signal.butter(4, np.array([f1Plot,f2Plot])/(fs/2), 'bandpass')
            
            y1_filt=filtfilt(b, a, y1)
            y1_r_filt=filtfilt(b, a, y1_r)
            y2_filt=filtfilt(b, a, y2)
            
            
            plt.figure()     
            ax1=plt.subplot(311)
            ax1.plot(y1_filt/np.max(np.abs(y1_filt)),'-g',label='y1')
            ax1.plot(y2_filt/np.max(np.abs(y2_filt)),label='y2')
            ax1.plot(y1_r_filt/np.max(np.abs(y1_filt)),label='y1-noise')
            plt.xlabel('time')
            ax1.legend()
            ax1b=ax1.twinx()
            #ax1b.plot(torque,label='torque')
            ax1b.legend()
            plt.title('Depth is %5.2f ft'%(3.28*depthArray[iwin]))

            
            ax2=plt.subplot(312)
            ax2.plot(lag,ccf,label='ccf')
            ax2.plot(lag,ccf_r,label='ccf_processed')
            ax2.legend()
            
            plt.xlim(-5,5)
            plt.xlabel('lab time (sec)')
            ax3=plt.subplot(313)
            ax3.semilogx(vec_f,coh,'-r',label='coherence')
            #            ax3.plot(vec_f,coh,'-r',label='coherence')
            plt.legend()
            ax3b=ax3.twinx()
            ax3b.loglog(vec_f,spec1,label='spectrum of y1')
            ax3b.loglog(vec_f,spec2,label='spectrum of y2')
            #            ax3b.loglog(vec_f,spec1_r,label='spectrum of y1 after removing noise')
            ax3b.loglog(vec_f,spec1_r,label='spectrum of y1 after removing noise')
            #            ax3b.semilogx(vec_f,np.angle(trans),label='phase')
            
            ax3b.legend()
    else:
        transList.append(vec_f*0)
        transList_r.append(vec_f*0)
        cohList.append(vec_f*0)
        spec1List.append(vec_f*0)
        spec2List.append(vec_f*0)

#        transList.append(trans)
#organize transfer function
mtx_resp=np.zeros((nwin,(len(vec_f)-1)*2),dtype=complex)
mtx_transfer=np.zeros((nwin,len(vec_f)),dtype=complex)
mtx_coh=np.zeros((nwin,len(vec_f)))
mtx_spec1=np.zeros((nwin,len(vec_f)),dtype=complex)
mtx_spec2=np.zeros((nwin,len(vec_f)),dtype=complex)

    
freqwin=(vec_f/f1Plot)**4/(1+(vec_f/f1Plot)**4)*(1/(1+(vec_f/f2Plot)**4))

for iwin in range(nwin):
#    filter transer function in desired band
    trans_window=transList_r[iwin]*freqwin 
#    trans_ifft=np.concatenate((transList[iwin],np.conjugate(np.flipud(transList[iwin][1:-1]))))
    trans_ifft=np.concatenate((trans_window,np.conjugate(np.flipud(trans_window[1:-1]))))

    mtx_resp[iwin,:]=fftshift(ifft(trans_ifft))
    mtx_transfer[iwin,:]=transList[iwin]
    mtx_coh[iwin,:]=cohList[iwin]
    mtx_spec1[iwin,:]=spec1List[iwin]
    mtx_spec2[iwin,:]=spec2List[iwin]
    

    #%% test ifft
N=1000
vec_t=np.linspace(0,N*dt,N)

# two singal with constant phase shift
y1=getRicker(5,vec_t,t0=2) + np.random.rand(len(vec_t))*0.001
y2=getRicker(5,vec_t,t0=3) + np.random.rand(len(vec_t))*0.001
vec_f,coh,trans,nstack=calc_coherence2(dt,y1,y2,500)
#_,spec1=calc_spec(dt,y1,1,ntSegment,outputType='power')
ccf,lag=calcCCFStack(y1,y2,ntSegment,dt=0.01,fmin=1,fmax=20,overlapPerc=0)        
#---------------
trans_ifft=np.concatenate((trans,np.conjugate(np.flipud(trans[1:-1]))))

resp=fftshift(ifft(trans))  # response function in time domain, size=nfft

plt.figure()     
ax1=plt.subplot(311)
ax1.plot(y1,label='y1.')
ax1.plot(y2,label='y2')
ax1.plot(np.abs(resp),label='response')
ax1.plot(np.real(resp),label='response')


plt.xlabel('time')
ax1.legend()
ax2=plt.subplot(312)
ax2.plot(lag,ccf,label='ccf')
ax2.plot(lag,ccf,label='ccf')

plt.xlim(-10,10)
plt.xlabel('lag time (sec)')
ax3=plt.subplot(313)
#            ax3.semilogx(vec_f,coh,'-r',label='coherence')
ax3.plot(vec_f,coh,'-r',label='coherence')

plt.legend()

ax3b=ax3.twinx()
ax3b.plot(vec_f,np.angle(trans),label='phase')
#ax3b.plot(vec_f,np.abs(trans),label='phase')
#ax3b.plot(vec_f,np.abs(spec1),label='phase')

plt.legend()
plt.xlabel('frequency')
plt.show()



#%% test remove coherence noise
depthMin=10100/3.28;
depthMax=10130/3.28;
maskcut= (data[holedepthName]>depthMin) & (data[holedepthName]< depthMax)
#mask= (data['01_Global.Pason_Surface.holeDepth']>depthMin) & (data['01_Global.Pason_Surface.holeDepth']< depthMax)
datacut = data.loc[maskcut]
#assign channels
torque=datacut['01_GLOBAL_PASON_TTS-STRQ']
tension=datacut['01_GLOBAL_PASON_TTS-TENSION']
drpm=datacut['01_GLOBAL_PASON_TTSV-DRPM']
holedepth = datacut[device+'-HOLEDEPTH']
rigstate=datacut[device+'-copRigState']
bitdepth=datacut[device+'-BIT_DEPTH']
srpm=datacut[device+'-SRPM']
pressure=datacut['01_GLOBAL_PASON_TTS-PRESSURE']

ar_tts=datacut['01_GLOBAL_PASON_TTS-RAD_ACCELERATION']
at_tts=datacut['01_GLOBAL_PASON_TTS-TAN_ACCELERATION']
az_tts=datacut['01_GLOBAL_PASON_TTS-AXIAL_ACCELERATION']
ah_tts=np.sqrt(ar_tts**2+at_tts**2)

latVibration=emsLatAccelT_aligned.loc[maskcut]*9.8
ems_gz=ems_gz_aligned.loc[maskcut]*9.8
ems_gx=ems_gx_aligned.loc[maskcut]*9.8
ems_gy=ems_gy_aligned.loc[maskcut]*9.8
ah_ems=np.sqrt(ems_gx**2+ems_gy**2)

#%% Plot  acceletation
fignum=np.int(np.random.rand()*100)  
plt.figure(fignum)

ax1=plt.subplot(311)
#plt.plot(wob.div(4448.2).resample('1s').mean(),'-g',label='RIG WOB')
plt.plot(torque.div(1350).resample('100ms').mean(),linewidth=1,label='Surface torque (kft-lb)')

#plt.plot(tension/10000/4.448,label='tension (10000 lb)')
ax1.grid(color='k', linestyle='--', linewidth=0.1)

plt.legend()
plt.show()
plt.ylabel('torque',fontsize=15)    
plt.xlabel('Time',fontsize=15)
#plt.ylim(-4,30)

ax1b=ax1.twinx()
ax1b.plot(srpm.resample('100ms').mean(),'-r',label='Surface RPM')


ax2=plt.subplot(312,sharex=ax1)
#plt.plot(latVibration,linewidth=1,label='Downhole Acceleration Amplitude')
plt.plot(ems_gx,linewidth=1,label='Downhole ax')
plt.plot(ems_gy,linewidth=1,label='Downhole ay')
plt.plot(ems_gz,linewidth=1,label='Downhole az')

plt.legend()
plt.show()
ax2.set_ylabel('Acceleration (m/s2)',fontsize=15)
plt.xlabel('Time',fontsize=15)
#plt.ylim(-20,20)


#ax2b=ax2.twinx()
#ax2b.grid(color='k', linestyle='--', linewidth=0.1)  


ax3=plt.subplot(313,sharex=ax1)
plt.plot(ar_tts,linewidth=1,label='Surface ar')
plt.plot(at_tts,linewidth=1,label='Surface at')
plt.plot(az_tts,linewidth=1,label='Surface az')

#plt.plot(emsRPM_10Hz,linewidth=1,label='BHA RPM')
#plt.ylim(-100,350)
plt.legend()
plt.ylabel('Surface acceleration',fontsize=15)
#ax3b.yaxis.label.set_color('red')
#plt.xlabel('Time',fontsize=15)
#plt.show()
#%% remove noise
y1=ah_tts
y2=ah_ems

fs=100
dt=1/fs
# number of samples within each long window (need to be short so that relationship is constant)
ntwin=y1.shape[0]
# number of samples within each short windows (long enough to cover possible delays)
ntSegment=10*fs 


#calculate raw spectrum
vec_f,spec1=calc_spec(dt,y1,ntwin//ntSegment,ntSegment,outputType='power')
_,spec2=calc_spec(dt,y2,ntwin//ntSegment,ntSegment,outputType='power')
_,spec_torque=calc_spec(dt,torque,ntwin//ntSegment,ntSegment,outputType='power')
_,spec_tension=calc_spec(dt,tension,ntwin//ntSegment,ntSegment,outputType='power')
_,spec_pressure=calc_spec(dt,pressure,ntwin//ntSegment,ntSegment,outputType='power')

#Calculate coherence
vec_f,coh_12,_,_=calc_coherence2(dt,y1,y2,ntSegment)
vec_f,coh_1_ah,_,_=calc_coherence2(dt,y1,ah_tts,ntSegment)
vec_f,coh_1_at,_,_=calc_coherence2(dt,y1,at_tts,ntSegment)
vec_f,coh_1_ar,_,_=calc_coherence2(dt,y1,ar_tts,ntSegment)
vec_f,coh_1_pressure,_,_=calc_coherence2(dt,y1,pressure,ntSegment)
vec_f,coh_1_tension,trans,nstack=calc_coherence2(dt,y1,tension,ntSegment)
vec_f,coh_1_torque,_,nstack=calc_coherence2(dt,y1,torque,ntSegment)

#frequency band to plot
f1Plot=2
f2Plot=10
#frequency band to remove noise
fmin=0.1
fmax=45

#        pdb.set_trace()        
#y1_12,y1_r=remove_coh_signal4(torque,az_tts,ar_tts,y1,dt,ntSegment,
#                            fmin,fmax,overlapPerc=0.0,detrend=False)

_,y1_r=remove_coh_signal4(torque,pressure,az_tts,y1,dt,ntSegment,
                            fmin,fmax,overlapPerc=0.0,detrend=False)

_,y2_r=remove_coh_signal4(torque,pressure,az_tts,y2,dt,ntSegment,
                            fmin,fmax,overlapPerc=0.0,detrend=False)

#y1_r,spec1_r=remove_coh_signal2(pressure,y1,dt,ntSegment,
#                            fmin,fmax,overlapPerc=0.0,detrend=False)

#y2_r,spec2_r=remove_coh_signal2(pressure,y2,dt,ntSegment,
#                            fmin,fmax,overlapPerc=0.0,detrend=False)
#calculate cross-correlation
ccf_r,lag=calcCCFStack(y1_r,y2_r,ntSegment,dt=0.01,fmin=f1Plot,fmax=f2Plot,overlapPerc=0) 
ccf,lag=calcCCFStack(y1,y2,ntSegment,dt=0.01,fmin=f1Plot,fmax=f2Plot,overlapPerc=0)        

#        ccf,lag=calcCCFStack(y1,y2,ntSegment,dt=0.01,fmin=f1,fmax=f2,overlapPerc=0,normalize=False)        


_,spec1_r=calc_spec(dt,y1_r,ntwin//ntSegment,ntSegment,outputType='power')
#_,spec1_r=calc_spec(dt,y1_r*(sp.signal.hamming(len(y1_r))),ntwin//ntSegment,ntSegment,outputType='power')

vec_f,coh_12_r,_,_=calc_coherence2(dt,y1_r,y2,ntSegment)
   

#calculate spectra after denoise
#_,spec1_r=calc_spec(dt,y1_r,ntwin//ntSegment,ntSegment,outputType='power')
#_,spec2_r=calc_spec(dt,y2_r,ntwin//ntSegment,ntSegment,outputType='power')

#calculate coherence after denoise
vec_f,coh_1_ah_r,_,_=calc_coherence2(dt,y1_r,ah_tts,ntSegment)
vec_f,coh_1_at_r,_,_=calc_coherence2(dt,y1_r,at_tts,ntSegment)
vec_f,coh_1_ar_r,_,_=calc_coherence2(dt,y1_r,ar_tts,ntSegment)
vec_f,coh_1_pressure_r,_,_=calc_coherence2(dt,y1_r,pressure,ntSegment)
vec_f,coh_1_tension_r,trans,nstack=calc_coherence2(dt,y1_r,tension,ntSegment)
vec_f,coh_1_torque_r,_,nstack=calc_coherence2(dt,y1_r,torque,ntSegment)

#Filter results

b, a = signal.butter(4, np.array([f1Plot,f2Plot])/(fs/2), 'bandpass')

y1_filt=filtfilt(b, a, y1)
y1_r_filt=filtfilt(b, a, y1_r)
y2_filt=filtfilt(b, a, y2)
pressure_filt=filtfilt(b, a, pressure)

#Plot time series

plt.figure()     
ax1=plt.subplot(311)
ax1.plot(y1_filt/np.max(np.abs(y1_filt)),'-g',label='y1')
ax1.plot(2+y2_filt/np.max(np.abs(y2_filt)),label='y2')
ax1.plot(y1_r_filt/np.max(np.abs(y1_filt)),label='y1-noise')
plt.xlabel('time')
ax1.legend()



ax2=plt.subplot(312)
ax2.plot(lag,ccf,label='ccf')
ax2.plot(lag,ccf_r,label='ccf_processed')
ax2.legend()

plt.xlim(-5,5)
plt.xlabel('lab time (sec)')
ax3=plt.subplot(313)
#ax3.semilogx(vec_f,coh_12,'-r',label='coherence')
#            ax3.plot(vec_f,coh,'-r',label='coherence')
#plt.legend()
ax3b=ax3.twinx()
ax3b.loglog(vec_f,spec1,label='spectrum of y1')
#ax3b.loglog(vec_f,spec2,label='spectrum of y2')
ax3b.loglog(vec_f,spec_pressure,label='pressure')
ax3b.loglog(vec_f,spec1_r,label='spectrum of y1 after removing noise')
#ax3b.loglog(vec_f,spec1_r,label='spectrum of y1 after removing noise')
#            ax3b.semilogx(vec_f,np.angle(trans),label='phase')

ax3b.legend()

plt.suptitle('Spectrum',fontsize=20)

plt.figure()     
ax1=plt.subplot(311)
#ax1.semilogx(vec_f,coh_1_ah,label='coherence 1-ah')
#ax1.semilogx(vec_f,coh_1_at,label='coherence 1-at')
#ax1.semilogx(vec_f,coh_1_ar,label='coherence 1-ar')
#ax1.semilogx(vec_f,coh_1_torque,label='coherence 1-torque')
#ax1.semilogx(vec_f,coh_1_tension,label='coherence 1-tension')
ax1.semilogx(vec_f,coh_1_pressure,label='coherence 1-pressure')
plt.title('Coherence',fontsize=15)    
plt.ylabel('Coherence',fontsize=15)    
#plt.xlabel('Time',fontsize=15)
plt.legend()
ax2=plt.subplot(312,sharex=ax1)
ax2.semilogx(vec_f,coh_12,label='coherence 12')
#ax2.semilogx(vec_f,coh_12_r,label='coherence 12_processed')

plt.legend()
plt.ylabel('Coherence',fontsize=15)    

ax3=plt.subplot(313,sharex=ax1)
ax3.loglog(vec_f,spec1,label='y1')
ax3.loglog(vec_f,spec2,label='y2')
#            ax3b.loglog(vec_f,spec1_r,label='spectrum of y1 after removing noise')
#ax3.loglog(vec_f,spec1_r,label='y1 after removing noise')
#ax3.loglog(vec_f,spec_torque,label='torque')
ax3.loglog(vec_f,spec_tension,label='tension')
ax3.loglog(vec_f,spec_pressure,label='pressure')
plt.xlabel('Frequency / hz',fontsize=15)
plt.ylabel('Power Spectrum',fontsize=15)    

#            ax3b.semilogx(vec_f,np.angle(trans),label='phase')

ax3.legend()




#%% Plot  spectra

plt.figure()     
plt.suptitle('Depth = %5.0f ft'%(depthMax*3.281))
ax1=plt.subplot(211)
#ax1.semilogx(vec_f,coh_1_ah,label='coherence 1-ah')
#ax1.semilogx(vec_f,coh_1_at,label='coherence 1-at')
#ax1.semilogx(vec_f,coh_1_ar,label='coherence 1-ar')
#ax1.semilogx(vec_f,coh_1_torque,label='coherence 1-torque')
#ax1.semilogx(vec_f,coh_1_tension,label='coherence 1-tension')
ax1.semilogx(vec_f,coh_1_pressure,label='coherence 1-pressure')
ax1.semilogx(vec_f,coh_12,label='coherence 12')
#ax1.set_xlabel('Frequency', fontsize=20)
ax1.set_ylabel('Coherence', fontsize=20)

#ax1.semilogx(vec_f,coh_12_r,label='coherence 12_processed')
plt.legend()
ax2=plt.subplot(212,sharex=ax1)
ax2.loglog(vec_f,spec1,label='y1')
ax2.loglog(vec_f,spec2,label='y2')
ax2.loglog(vec_f,spec1_r,label='y1 - noise')

#ax2.loglog(vec_f,spec_torque,label='torque')
#ax2.loglog(vec_f,spec_tension,label='tension')
ax2.loglog(vec_f,spec_pressure,label='pressure')
ax2.set_xlabel('Frequency', fontsize=20)
ax2.set_ylabel('Power Spectra', fontsize=20)

#ax3b.semilogx(vec_f,np.angle(trans),label='phase')

ax2.legend()
ax1.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)
#%% Plot  time series
plt.figure()     
ax1=plt.subplot(311)
ax1.plot(pressure_filt,label='pressure')
#ax1.plot(pressure,label='pressure')

#ax1.plot(y1_r_filt/np.max(np.abs(y1_filt)),label='y1-noise')
plt.xlabel('time')
ax1.legend()

ax2=plt.subplot(312,sharex=ax1)
ax2.plot(y2_filt,label='y2')
#ax2.plot(y2,label='y2')

ax2.legend()

ax3=plt.subplot(313,sharex=ax1)
ax3.plot(y1_filt,label='y1')
#ax3.plot(y1,label='y1')

#plt.ylim(-20000,20000)
ax3.plot(y1_r_filt,label='y1 remove noise')
#ax3.plot(y1_r,label='y1 remove noise')

ax3.legend()

    #%% debug remove noise tool


