%parameters settings for calculating OBS orientations from noise


%% data parameters
% Specify the sampling rate of the input data
dt=1.0;
% Channel name of Z component:
chz='LHZ';
polarity=1; % for z data ,1 if up is positive, -1 if down is positive;
ifLH=0; % 1 if H1 and H2 are in GSN convention, H2 90 degress clockwise from H1. 0 otherwise 
% sta0 coord

ch1=[chz(1),'H1'];
ch2=[chz(1),'H2'];

%datadir='/Volumes/Work/Lau_research/OBS_data/';
% ------------directory of SAC data-------------------
%datadir='/Volumes/LauData_YangZha/WHO-antelope/WHO-STAID/';
rootdir=pwd;

%datadir='/Volumes/Work/Lau_research/orientation/tool/test_data/'
datadir=[rootdir,'/test_data/'];



% THERE ARE TWO STATION LISTS, THE FIRST ONE IS "CENTERSTATIONLIST":
% STATIONS TO CALCULATE ORIENTATION FOR
% THE SECOND ONE IS "STATIONLIST": STATIONS TO SERVE AS VIRTUAL SOURCES 
% WE LOOP THROUGH BOTH

%centerstalist= '/Volumes/Work/Lau_research/orientation/ccf/stationlist_whoi.dat'
% stations to calculate ccf with each center station as virtural sources
%stationlist = '/Volumes/Work/Lau_research/orientation/ccf/stationlist_whoi.dat'
%centerstalist= '/Volumes/Work/Lau_research/orientation/tool/testlist.dat';
centerstalist= [rootdir,'/testlist.dat'];
% 
%stationlist = '/Volumes/Work/Lau_research/orientation/ccf/stationlist_all.dat'
%stationlist= '/Volumes/Work/Lau_research//orientation/tool/testlist.dat';
stationlist= [rootdir,'/testlist.dat'];
%stationlist = '/Volumes/Work/Lau_research/orientation/ccf/twostation.dat'

v_sta0=textread(centerstalist,'%s');
v_sta=textread(stationlist,'%s');


%% settings for calculating cross-term correlations

% ------------root directory to storing CCF------------
%ccfdir='/Volumes/Work/Lau_research/orientation/tool/testccf/';
ccfdir=[rootdir,'/testccf/'];
% specify the largest station distances in the network
maxdist=500;
% calculate the maximum time lag based on maximun station spacings
maxlag=maxdist/dt;




year=2010;
jdaystart=1;
jdayend=10;

dist_min=60; % minimum distance to calculate CCF in km


%% =========settings for estimating orientations
T1=10;T2=20; % frequency band to use, try to avoid Schotte wave band which has prograde particle motion
% range of group velocity to use for windoing CCF in km/s
cmax=5.0;
cmin=2.5;
% station to calculate orientation for

sta1 = 'A02W';
% -----QC parameters------------
C_cutoff=0.5;
S_cutoff=0.;
SNR_cutoff=2;

phase_cutoff=180;
data_cut=1;% 1: for throwing out data 
confidence=0.05; % 0.32 1sigma 0.05 2 sigma 0.0025 3 sigma

% ----plotting, 1 for true-----------
plot_result=1; 
plot_coverage=1;
plot_QC=1;
plot_uncertainty=1;
manualQC=0;
%----save result
ifsave=1;
