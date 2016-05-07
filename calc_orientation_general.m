% Calculate OBS orientation from ambient noise cross correlation record for single stationpair
% Evaluate Coherence of R_zz and R_rz in spectral domain
% input: CCF of R_ZZ R_1Z and R_2Z 
% 		 locations of both stations
% output : Srz (correlation coefficient of Hiltbert(C_zz) and C_rr), normalized by Szz
%
% Procedures:
% 		1. setup parameters
% 		2. read CCF, C_ZZ,C_NZ,C_EZ;
% 		3. hilbert transform C_ZZ;
% 		4. loop through phi and calculate C_RZ;
% 		5. for each phi, calculate Cxy(f)=crossspectrum of R_zz and R_rz;
%		6. determine phi corresponding to maximum Cxy(f0) or avg(Cxy) and calculate orientation;
% 		
%
% -------------------station settings-------------------
clear


% dt=1
% polarity=1; % for z data ,1 if up is positive, -1 if down is positive;
% T1=15;T2=30; % frequency band to use
% % range of group velocity to use
% cmax=5; S
% cmin=2.5;
% 
% maxlag=500/dt;
setting_tool;
sta1 = 'G05B';
sta2 = 'G11B';
% %sta2 = 'A02W';
% 
% datadir='/Volumes/Work/Lau_research/OBS_data/';
% ccfdir=['/Volumes/Work/Lau_research/orientation/ccf/',sta1,'/'];
dir0=pwd;


%% ===========SHOULD NOT NEED TO CHANGE ANYTHING BELOW===================
sta1dir=[ccfdir,sta1,'/']; % dir to save all cross terms about this central station
cd(sta1dir)
[vpair,lat1_vec,lon1_vec,dep1_vec,lat2_vec,lon2_vec,dep2_vec,d_vec,az_vec,baz_vec]...
    =textread('stationpair.txt','%s %f %f %f %f %f %f %f %f %f','headerlines',1);

lat1=lat1_vec(1);
lon1=lon1_vec(1);
dep1=dep1_vec(1);

for ip=1:length(vpair)
    if(strfind(vpair{ip},sta2))
        ip0=ip;
        lat2=lat2_vec(ip);
        lon2=lon2_vec(ip);
        dep2=dep2_vec(ip);
        dist=d_vec(ip);
        az=az_vec(ip);
        break
    end
    
end

filenamez=['Xcorr_',chz,chz,'.dat'];
filename1=['Xcorr_',ch1,chz,'.dat'];
filename2=['Xcorr_',ch2,chz,'.dat'];
% -------------------read station coord -------------------
% % sta1
% cd([datadir,sta1]);
% [name1,lat1,lon1,dep1]=textread('coord.dat','%s %f %f %f');% station coordinate 
% 
% % sta2
% 
% cd([datadir,sta2]);
% [name2,lat2,lon2,dep2]=textread('coord.dat','%s %f %f %f');% station coordinate 

%Z1=imag(hilbert(E1));
% 
% [delta,az]=distance(lat1,lon1,lat2,lon2);
% dist=delta*6371*pi/180;

%------------------- read CCF------------------------
%cd([ccfdir,sta1,'_',sta2]);
cd([sta1dir,sta1,'_',sta2]);

 %------------------- read CCF------------------------
    
    if~(exist([sta1dir,sta1,'_',sta2]))
        display('no data')
        
    end
    

    
    cd([sta1dir,sta1,'_',sta2]);
    %
    
    temp=load(filenamez);
    tlag=temp(:,1);
    R_zz=temp(:,2); % take the symetric component
    
    temp=load(filename1);
    R_1z=temp(:,2);
    
    temp=load(filename2);
    R_2z=temp(:,2);
    
    %swap H1 and H2 for non-GSN convention;
     	if(ifLH)
    		Rtemp=R_2z;
    		R_2z=R_1z;
    		R_1z=Rtemp;
    		display('swap H1 and H2 channels');
       end

%------------filter and hilbert transform--------

fn=1/2/dt;
[b,a]=butter(4,[1/fn/T2,1/fn/T1]);
FR_zz0=filtfilt(b,a,R_zz);
FR_1z0=filtfilt(b,a,R_1z);
FR_2z0=filtfilt(b,a,R_2z);

%HR_zz=-imag(hilbert(FR_zz)); % shift R_zz ahead 90 degrees
HR_zz0=polarity*imag(hilbert(FR_zz0)); % shift R_zz -90 degrees
%HR_zz=polarity*imag(hilbert(R_zz)); % shift R_zz -90 degrees


%--------CUT to phase arrival time-------------------------------
	win_signal=[dist/cmax dist/cmin];
	win_noise = win_signal+100;
	indcut=(tlag>win_signal(1)).*(tlag<win_signal(2));
%	indcut=(tlag>-80).*(tlag<-40);
	indnoise=(tlag>win_noise(1)).*(tlag<win_noise(2));

	FR_zz=FR_zz0(find(indcut));
	tlag_win=tlag(find(indcut));	
	FR_1z=FR_1z0(find(indcut));
	FR_2z=FR_2z0(find(indcut));	
	HR_zz=HR_zz0(find(indcut));
	
% 
%-------LOOP THROUGH PHI ------------------------------
%
Szz=xcorr(HR_zz,HR_zz,0);
v_phi=1:360; % vector of orientation vector phi is angle COUNTER CLOCKWISE from E to H1
figure;hold on
set(gcf,'position',[100 300 650 400])
%plot(tlag,FR_zz0,'-k','linewidth',1);
plot(tlag,HR_zz0,'-k','linewidth',2);
	
%	nfft=2^nextpow2(length(HR_zz));
	
for k=1:length(v_phi)
	theta=90-az-v_phi(k); % angle COUNTER CLOCKWISE from H1 to R
	[FR_rz,FR_tz] = rotate_vector(FR_1z,FR_2z,theta);
	Srz=xcorr(FR_rz,HR_zz,0); % correlation of Crz and Hilbert(Czz)
	Srr=xcorr(FR_rz,FR_rz,0); 
	HSrz=xcorr(FR_rz,FR_zz,0); % correlation of Crz and Czz

	S(k)=Srz/Szz;		
	H(k)=HSrz/Szz;
	C(k)=Srz/sqrt(Szz*Srr);		
	P(k)=sqrt(S(k)^2+H(k)^2); % power of correlation

%	%---calculate cross spectral density
%	[Prz,vec_f] = cpsd(FR_rz,HR_zz,[],[],nfft,1);
%	indf=find((vec_f<0.2).*(vec_f>0.05)); % aveage coherence from 5 to 20 s
%	P(k)=sum(abs(Prz(indf)));
%	realP(k)=sum(real(Prz(indf)));

%	plot(tlag_win,FR_rz,'-r');	drawnow
end
	
% 	search for max correlation between Crz and 	H(Czz)
	[Smax,imax]=max(S)
%	[Pmax,imax]=max(P.*(S>0));
	phase=atan2(H(imax),S(imax))*180/pi		

	phi0=v_phi(imax);
%
%	[ind0,zero,nzero,sign]=FindArrayZero(v_phi,S,[min(v_phi) max(v_phi)]); % find angles with zero correlation
%	if(nzero==2)
%		if(sign(1)>0) 
%			phi1=(zero(1)+zero(2))/2;
%		elseif(sign(2>0))
%			phi1=(zero(1)+zero(2))/2+180;
%		end
%	
%	end
	display(['angle between H1 and North is: ', num2str(phi0)]);
%	display(['angle determined from T direction is ', num2str(phi1)]);	

	[FR_rz0,FR_tz0] = rotate_vector(FR_1z0,FR_2z0,90-az-phi0);
	%calculating SNR
	Amp_S=max(abs(FR_rz0(find(indcut))));
	Amp_N=std(FR_rz0(find(indnoise)));
	SNR=Amp_S/Amp_N;
	
	Amp_SZ=max(abs(FR_zz0(find(indcut))));
	Amp_NZ=std(FR_zz0(find(indnoise)));
	SNR_Z=Amp_SZ/Amp_NZ;

%	plot(tlag,HR_zz,'-m');
	plot(tlag,FR_rz0,'-r','linewidth',2);
%	plot(tlag,FR_tz0,'-c','linewidth',1);
	
	title([sta1,'-',sta2,'-ZZ']);
%	xlim([min(tlag_win),max(tlag_win)]);
	yl=get(gca,'ylim');
	plot([win_signal(1) win_signal(1)],yl,'-k');
	plot([win_signal(2) win_signal(2)],yl,'-k');

	ylabel('CCF','fontsize',15)	;
	xlabel('Timelag / seconds','fontsize',15);
	xlim([0 200]);

	l=legend('H_{ZZ}','C_{RZ}');
	set(l,'fontsize',12)

	% PLOT waveform
	if(1)
		figure;
		subplot(3,1,1);hold on;
		title([sta1,'-',sta2,'-ZZ']);	
		plot(tlag,FR_zz0,'-k','linewidth',1.5);
		xlim([0,200]);
		ylim([-max(abs(FR_zz0)) max(abs(FR_zz0))]);

		l=ylabel('C_{zz}'); set(l,'fontsize',18);
%		xlabel('Timelag / seconds','fontsize',15);
		plot([win_signal(1) win_signal(1)],[-max(abs(FR_zz0)) max(abs(FR_zz0))],'-k');
		plot([win_signal(2) win_signal(2)],[-max(abs(FR_zz0)) max(abs(FR_zz0))],'-k');
	
		subplot(3,1,2);

		plot(tlag,FR_rz0,'-k','linewidth',1.5);hold on
        h=plot(tlag,HR_zz0,'--r','linewidth',1.5);hold on		
        %legend(h,'H_{ZZ}');
        
%		set(h,'color',[0.5 0.5 0.5]);

		plot([win_signal(1) win_signal(1)],[-max(abs(FR_zz0)) max(abs(FR_zz0))],'-k');
		plot([win_signal(2) win_signal(2)],[-max(abs(FR_zz0)) max(abs(FR_zz0))],'-k');

	%	plot([win_noise(1) win_noise(1)],[-Amp_S Amp_S],'-r');
	%	plot([win_noise(2) win_noise(2)],[-Amp_S Amp_S],'-r');

		xlim([0,200]);
		ylim([-max(abs(FR_zz0)) max(abs(FR_zz0))]);

		
		
		l=ylabel('C_{rz}'); set(l,'fontsize',18);

%		xlabel('Timelag / seconds','fontsize',15);

	
		subplot(3,1,3);hold on;
		plot(tlag,FR_tz0,'-k','linewidth',1.5);
		xlim([0,200]);
		ylim([-max(abs(FR_zz0)) max(abs(FR_zz0))]);

	
	
		l=ylabel('C_{tz}'); set(l,'fontsize',18);
		plot([win_signal(1) win_signal(1)],[-max(abs(FR_zz0)) max(abs(FR_zz0))],'-k');
		plot([win_signal(2) win_signal(2)],[-max(abs(FR_zz0)) max(abs(FR_zz0))],'-k');

		xlabel('Timelag / seconds','fontsize',18);
	
		set(gcf,'position',[100,300,700,500]);
	end


	figure; hold on;
	plot(v_phi,S,'-r','linewidth',2);
	plot(v_phi,C,'-b','linewidth',2);
%	plot(v_phi,H,'-bo');	
%	plot(v_phi,P,'-ko');	
	plot(v_phi,v_phi*0,'-k');
	plot([phi0 phi0],[-1 1],'--k');
	text(phi0-60,0.8,['\psi_0= ',num2str(phi0),'\circ'],'fontsize',15)

	l2=legend('S_{rz}','R_{rz}');	
	set(l2,'fontsize',15,'location','best');
	xlabel('Orientation angle \psi (degrees)','fontsize',18);
	ylabel('Correlation coefficient','fontsize',18);
	title([sta1,'-',sta2,'-ZZ']);	
	set(gcf,'position',[100 300 700 400])
%------------------- loop through phi to calculate S(phi)------------------------
	cd(dir0)
