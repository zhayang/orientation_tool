%parameters settings for calculating OBS orientations from noise


%% data parameters
% Specify the sampling rate (in seconds) of the input data
dt=1.0;
% Channel name of Z component:
chz='LHZ';
polarity=1; % for z data ,1 if up is positive, -1 if down is positive;
ifLH=0; % 1 if H1 and H2 are in GSN convention, H2 90 degress clockwise from H1. 0 otherwise 
% sta0 coord

% channel names for H1 and H2
ch1=[chz(1),'H1'];
ch2=[chz(1),'H2'];


% root directory of your project
rootdir=pwd;

datadir=[rootdir,'/test_data/'];



% there are two station lists, the first one is "centerstationlist":
% stations to calculate orientation for
% the second one is "stationlist": stations to serve as virtual sources 
% we loop through both

centerstalist= [rootdir,'/testlist.dat'];
stationlist= [rootdir,'/testlist.dat'];

% read in station list
v_sta0=textread(centerstalist,'%s');
v_sta=textread(stationlist,'%s');


%% settings for calculating cross-term correlations

% ------------root directory to storing CCF------------
ccfdir=[rootdir,'/testccf/'];
% specify the largest station distances (in km) in the network
maxdist=500;
% calculate the maximum time lag based on maximun station spacings
maxlag=maxdist/dt;



% specify the year and the dates of the data to be processed
year=2010;
jdaystart=1;
jdayend=10;

dist_min=60; % minimum distance to calculate CCF in km


%% =========settings for estimating orientations
T1=10;T2=20; % frequency band to use in terms of period (S), try to avoid Schotte wave band which has prograde particle motion. You want to test multiple sets of this parameter to get the most energetic band for ambient noise.

% range of group velocity to use for windoing the cross-correlation functions in km/s
cmax=5.0;
cmin=2.5;
% station to calculate orientation for

sta1 = 'A10W';
% -----QC parameters------------

C_cutoff=0.0; % minimum cutoff correlation coefficient
S_cutoff=0.2; % minimum cutoff 
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
