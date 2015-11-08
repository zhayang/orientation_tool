% Calculate OBS orientation from ambient noise cross correlation record from multiple stationpairs
% input: CCF of R_ZZ R_1Z and R_2Z between station0 and multiple other pairs
% 		 locations of both stations
% output : Srz (correlation coefficient of Hiltbert(C_zz) and C_rr), normalized by Szz for each station pair
%			phi_j for each station pair
%			mean,median and std of phi
% Procedures:
% 		1. setup parameters, load staionlist
% 		2. loop through station pairs, read CCF, C_ZZ,C_NZ,C_EZ;
% 		3. filter and hilbert transform C_ZZ;
% 		4. loop through phi and calculate C_RZ;
% 		5. for each phi, calculate S(phi)=Srz/Szz, H(phi)=HS_rz/Szz , P^2=(S^2+H^2) and plot;
%		6. determine phi corresponding to maximum P(phi) or S(phi) and calculate orientation;
%
%
%% -------------------station settings-------------------
clear
 dir0=pwd;
% dt=1;
% %ifQC=0;
% polarity=1; % for z data ,1 if up is positive, -1 if down is positive;
% T1=10;T2=20; % frequency band to use, try to avoid Schotte wave band which has prograde particle motion
%
% % range of group velocity to use for windoing CCF
% cmax=5.0;
% cmin=2.5;
% % -----QC parameters------------
% C_cutoff=0.5;
% S_cutoff=0.;
% SNR_cutoff=0;
%
% phase_cutoff=180;
% data_cut=0;% 1: for throwing out data
% confidence=0.05; % 0.32 1sigma 0.05 2 sigma 0.0025 3 sigma
%
% %-------------------------
% maxlag=500/dt;
%
%
% sta1 = 'A02W';
%
% %sta2 = 'A10W';
% %stationlist = '/Volumes/Work/Lau_research/orientation/ccf/testlist.dat'
% %v_sta=textread(stationlist,'%s');
% % sta1 coord
%
% %datadir='/Volumes/Work/Lau_research/OBS_data/';
% ccfdir='/Volumes/Work/Lau_research/orientation/tool/ccf/';

setting_tool;

%% ===========SHOULD NOT NEED TO CHANGE ANYTHING BELOW===================

sta1dir=[ccfdir,sta1,'/']; % dir to save all cross terms about this central station

%cd([datadir,sta1]);
%[name0,lat1,lon1,dep1]=textread('coord.dat','%s %f %f %f');% station coordinate

cd(sta1dir);
[vpair,lat1_vec,lon1_vec,dep1_vec,lat2_vec,lon2_vec,dep2_vec,d_vec,az_vec,baz_vec]...
    =textread('stationpair.txt','%s %f %f %f %f %f %f %f %f %f','headerlines',1);

lat1=lat1_vec(1);
lon1=lon1_vec(1);
dep1=dep1_vec(1);

filenamez=['Xcorr_',chz,chz,'.dat'];
filename1=['Xcorr_',ch1,chz,'.dat'];
filename2=['Xcorr_',ch2,chz,'.dat'];

phi0=nan*ones(size(vpair));
ind_pair=[]; % index of pairs used.
% loop through stations
ray=zeros(length(vpair),4)*nan;
for ip=1:length(vpair)
    clear R_zz R_1z R2z S C
    % -------------------read station coord -------------------
    % sta2
    pair=vpair{ip}
    %sta2=pair(6:9);
    sta2=v_sta{ip};
    
    lat2=lat2_vec(ip);
    lon2=lon2_vec(ip);
    dep2=dep2_vec(ip);
    
    [delta,az]=distance(lat1,lon1,lat2,lon2);
    dist=delta*6371*pi/180;
    
    %	if(dist<120)
    %		continue
    %	end
    %------------------- read CCF------------------------
    
    if~(exist([sta1dir,sta1,'_',sta2]))
        display('no data')
        continue
    end
    
    if(sta1(4)~=sta2(4))
        %		display('skip')
        %		continue
    end
    
    cd([sta1dir,sta1,'_',sta2]);
    %
    if~(exist(filenamez))
        continue
    end
    
    if~(exist(filename1))
        continue
    end
    if~(exist(filename2))
        continue
    end
    
    temp=load(filenamez);
    tlag=temp(:,1);
    R_zz=temp(:,2); % take the symetric component
    
    temp=load(filename1);
    R_1z=temp(:,2);
    
    temp=load(filename2);
    R_2z=temp(:,2);
    
    %swap H1 and H2 for GSN convention;
     	if(ifLH)
%    		Rtemp=R_2z;
%    		R_2z=R_1z;
%    		R_1z=Rtemp;
%    		display('swap H1 and H2 channels');
			R_2z=-R_2z;
				display('inverting H2 channel');

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
    
    % PLOT
    	if(0)
    		f2=figure;
    
    		subplot(3,1,1);
    		plot(tlag,FR_zz0);
    		xlim([-200,200]);
    %		title('R-ZZ');
    		title([sta1,'-',sta2,'-ZZ']);
    
    		subplot(3,1,2);
    		plot(tlag,FR_1z0);
    		xlim([-200,200]);
    
    
    		title('R-1Z');
    
    		subplot(3,1,3);
    		plot(tlag,FR_2z0);
    		xlim([-200,200]);
    
    
    		title('R-2Z');
    
    		set(gcf,'position',[100,300,900,900]);
    
    	end
    
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
    %------------------- loop through phi to calculate S(phi)------------------------
    
    %
    Szz=xcorr(HR_zz,HR_zz,0);
    v_phi=1:360; % vector of orientation vector phi is angle COUNTER CLOCKWISE from E to H1
    
    
    
    for k=1:length(v_phi)
%        theta=90-az-v_phi(k); % angle COUNTER CLOCKWISE from H1 to R
        theta=v_phi(k)-az; % angle COUNTER CLOCKWISE from H1 to R
		
        [FR_rz,FR_tz] = rotate_vector(FR_1z,FR_2z,theta);
        
        Srz=xcorr(FR_rz,HR_zz,0);
        Srr=xcorr(FR_rz,FR_rz,0);
        HSrz=xcorr(FR_rz,FR_zz,0); % correlation of Crz and Czz
        
        S(k)=Srz/Szz;
        %		H(k)=HSrz/Szz;
        H(k)=HSrz/xcorr(FR_zz,FR_zz,0);
        C(k)=Srz/sqrt(Szz*Srr);
        P(k)=sqrt(S(k)^2+H(k)^2); % power of correlation
        
        
        
    end
    
    %------------------------------------------------------------
    [Smax,imax]=max(S);
    %		[Pmax,ipmax]=max(P.*(S>0));
    phase(ip)=atan2(H(imax),S(imax))*180/pi;
    %------------------------------	---------------
    
    % apply a coherence cutoff for C.
    vec_S(ip)=Smax;
    vec_C(ip)=C(imax);
    phi_temp(ip)=v_phi(imax);
%    [FR_rz0,FR_tz0] = rotate_vector(FR_1z0,FR_2z0,90-az-v_phi(imax));
    [FR_rz0,FR_tz0] = rotate_vector(FR_1z0,FR_2z0,v_phi(imax)-az);
	
    %calculating SNR
    Amp_S=max(abs(FR_rz0(find(indcut))));
    Amp_N=std(FR_rz0(find(indnoise)));
    SNR_R(ip)=Amp_S/Amp_N;
    
    Amp_SZ=max(abs(FR_zz0(find(indcut))));
    Amp_NZ=std(FR_zz0(find(indnoise)));
    SNR_Z(ip)=Amp_SZ/Amp_NZ;
    
    % QC on waveform to decide if keep data
    	
    if(manualQC)
    		f1=figure;hold on
    		plot(tlag_win,FR_zz,'-b');
    		plot(tlag_win,HR_zz,'-k');
    
    	%	plot(tlag,HR_zz,'-m');
    		plot(tlag,FR_rz0,'-r');
    		title([sta1,'-',sta2,'SNR: ',num2str(SNR_R(ip))]);
    		legend('ZZ','H-ZZ','RZ');
    		xlim([min(tlag_win),max(tlag_win)]);
    
    %		figure; hold on;
    %		plot(v_phi,S,'-ro');
    %		plot(v_phi,H,'-bo');
    %		plot(v_phi,P,'-ko');
    %		plot(v_phi,v_phi*0,'-k');
    %
    		QC = input('DOES DATA QUALITY PASS? y/n [y]: ', 's');
    		if isempty(QC)
    		    QC = 'y';
    		end
    		close(f1);
    %		close(f2);
    	else
    		QC='y'; % if not qc, pass all results
    	end
    
    if((C(imax)>C_cutoff)&&(Smax>S_cutoff)&&(SNR_Z(ip)>SNR_cutoff)&&(SNR_R(ip)>SNR_cutoff)&&(abs(phase(ip))<phase_cutoff))
        phi0(ip)=v_phi(imax);
        ind_pair=[ind_pair;ip];
    end
    
    
    
    display(['angle between H1 and East is: ', num2str(phi0(ip)),' degrees']);
    %		display(['angle determined from T direction is ', num2str(phi1(ip))]);
    
    %
    ray(ip,:)=[lat1,lon1,lat2,lon2];
    
end

% --------some cirlular statistics on results---------
% using circular toolbox

phi=phi0(~isnan(phi0)); % taking out NANs

ndata=length(phi); % initial number of data

if(isempty(phi))
    display('no measurement satisfies data quality criteria')
    phi=nan;
end

rphi=circ_ang2rad(phi); % angle to radian

phi_med=circ_rad2ang(circ_median(rphi)); % median of phi
phi_mean=circ_rad2ang(circ_mean(rphi)); % mean of phi

% adjust to [0 360]
phi_med=phi_med+360*(phi_med<0);
phi_mean=phi_mean+360*(phi_mean<0);



%res_phi=phi-phi_med;

phi_std=circ_rad2ang(circ_std(rphi));
% standard deviation of sample

% using bootstrap to estimate standard deviation of mean angle
[v_mean mean_est std_mean] = bootstrap_circ(phi,10000);
std_mean

% 1-sigma confident level of mean , in degrees equivalent ot 95%
% confidence level
dconf = circ_rad2ang(circ_confmean(rphi,confidence))



% further taking out outliers
% retain only measurement within 95% interval
if( (data_cut)&&(~isnan(std_mean)) )
    
    % Data cut to reduce uncertanties
    % cut data outide of 95 confidence interval [mean-dconf mean+dconf].
    ind_phi=find((phi0<=phi_mean+2*std_mean).*(phi0>=phi_mean-2*std_mean))
    phi = phi0(ind_phi);
    ndata=length(phi);
    phi_mean=circ_rad2ang(circ_mean(circ_ang2rad(phi)));
    phi_med=circ_rad2ang(circ_median(circ_ang2rad(phi)));
    phi_std=circ_rad2ang(circ_std(circ_ang2rad(phi)));
    [v_mean mean_est std_mean] = bootstrap_circ(phi,10000);
    std_mean;
    dconf = circ_rad2ang(circ_confmean(circ_ang2rad(phi),confidence));
    % adjust to [0 360]
    phi_med=phi_med+360*(phi_med<0);
    phi_mean=phi_mean+360*(phi_mean<0);
    
end

%	res=phi-phi_med;%  residual
%	res0=res+(res<-180)*360-(res>180)*360; % shift everthing to [-180 180]
%	rres=abs(res).*(abs(res)<180)+(abs(res)>180).*(360-abs(res)); % adjust absolute residual to [0 180]
%	MAD=median(rres);
%	SMAD=1.482*MAD;

display(['mean of the orientation is ',num2str(phi_mean)]);
display(['std of the mean orientation is ',num2str(std_mean)]);
display(['median of the orientation is ',num2str(phi_med)]);
%display(['SMAD of the median orientation is ',num2str(SMAD)]);


%---------------PLOT-------------------------------------
if(plot_result)
    
    %plot angular coverage
    figure;
    
    %plot hist of mean from bootstrap
    %[theta_bin,n_bin] = rose(v_mean*pi/180,120);
    %h=rose(v_mean*pi/180,120);
    %phi=eq_data;
    
    [theta_bin,n_bin] = rose(phi*pi/180,30);
    %polar(0,max(n_bin),'-k');
    %polar(0,7);hold on
    
    %h=rose(phi*pi/180,30);
    %h=rose(phi*pi/180,30);hold on
    h=polar(theta_bin,n_bin); hold on
    x = get(h,'Xdata');
    y = get(h,'Ydata');
    g=patch(x,y,'c');
    %n_bin=15;
    h1=compass(max(n_bin)*cosd(phi_mean),max(n_bin)*sind(phi_mean),'-r');
        view([90 -90])
	
    %h2=compass(max(n_bin)*cosd(phi_eq),max(n_bin)*sind(phi_eq),'-k');
    %h1=compass(cosd(phi_mean),sind(phi_mean),'-r');
    %h2=compass(cosd(phi_eq),sind(phi_eq),'-k');
    set(h1,'linewidth',2);
    %set(h2,'linewidth',2);
    title(sta1,'fontsize',20);
    %set(gca,'xaxislocation','top');
    %cset(gca,'yaxislocation','right');
    %xlabel('N');ylabel('E');
    %tx=text(max(n_bin),max(n_bin),['\psi_{med}(SMAD) = ',num2str(phi_med),'(', num2str(SMAD),')'],'fontsize',13);
    tx2=text(max(n_bin)/1.5,max(n_bin),['\psi_{mean}(95%) = ',num2str(phi_mean),'(', num2str(std_mean*2),')'],'fontsize',13);
    %tx2=text(1/1.5,1,['\psi_{mean}(95%) = ',num2str(phi_mean),'(', num2str(std_mean*2),')'],'fontsize',13);
    tx3=text(max(n_bin)/1.5,max(n_bin)*0.8,['ndata = ',num2str(ndata)],'fontsize',13);
    %for ip=1:length(az_vec)
    %	polar([0,pi/2-(az_vec(ip)*pi/180)],[0,max(n_bin)])
    %end
    
    %lgd=legend([h1,h2],'Ambient noise','Earthquake','location','best');
    %set(lgd,'fontsize',15)
    
    set(gcf,'position',[680   560   740   400])
end

% left and right 95% confidence bounds
if(plot_uncertainty)
    % bootstrap phi_mean
    
    bins = [mean_est-40:0.5:mean_est+40]';
    Nb = length(bins);
    Db = bins(2)-bins(1);
    h = hist( v_mean, bins );
    norm = Db*sum(h);
    p_mu= h' / norm;
    
    C_mu = Db*cumsum(p_mu); % cumulative
    
    t=max(h);
    figure;
    set(gca,'LineWidth',2);
    hold on;
    hist(v_mean,bins);
    xlabel('theta, deg');
    ylabel('mean(theta)');
    xlim([phi_mean-40 phi_mean+40]);
    p0=plot( [mean_est, mean_est]', [0, t/5]', 'y-', 'LineWidth', 5);
    %plot( [mu_real, mu_real]', [0, t/5]', 'g-', 'LineWidth', 5);
    
    lb = bins(find( C_mu> 0.025, 1 ));
    rb = bins(find( C_mu> 0.975, 1 ));
    plot( [lb, lb]', [0, t/5]', 'r-', 'LineWidth', 5);
    p1=plot( [rb, rb]', [0, t/5]', 'r-', 'LineWidth', 5);
    
    lbx=phi_mean-2*std_mean;
    rbx=phi_mean+2*std_mean;
    plot( [lbx, lbx]', [0, t/5]', 'g-', 'LineWidth', 3);
    p2=plot( [rbx, rbx]', [0, t/5]', 'g-', 'LineWidth', 3);
    
    % 	lby=phi_mean-dconf;
    % 	rby=phi_mean+dconf;
    % 	plot( [lby, lby]', [0, t/5]', 'm-', 'LineWidth', 3);
    % 	p3=plot( [rby, rby]', [0, t/5]', 'm-', 'LineWidth', 3);
    
    % 	lbz=phi_mean-2*phi_std/sqrt(ndata);
    % 	rbz=phi_mean+2*phi_std/sqrt(ndata);
    % 	plot( [lbz, lbz]', [0, t/5]', 'c-', 'LineWidth', 3);
    % 	p4=plot( [rbz, rbz]', [0, t/5]', 'c-', 'LineWidth', 3);
    %
    legend([p0,p1,p2],'mean from boostap','empirical 95% confidence interval','mean+/- 2 \sigma');
    %legend([p0,p1,p2,p3 p4],'mean from boostap','empirical 95% confidence interval','mean+/- 2 \sigma','calculated using circ_ toolbox','+/- 2*sample \sigma /sqrt(ndata)');
    title(['sample size=',num2str(ndata),' \psi-mean = ',num2str(phi_mean),' \sigma-mean = ',num2str(std_mean)],'fontsize',15)
end

if(plot_QC)
    % phase and psi
    % figure;hold on
    % plot(phase,phi_temp,'o','markerfacecolor','b');
    % %plot(phase(SNR_Z>SNR_cutoff),phi_temp(SNR_Z>SNR_cutoff),'o','markerfacecolor','b');
    % xlabel('\Delta (\circ)','fontsize',15);ylabel('Orientation angle \psi(\circ)','fontsize',15);
    % title('Relative Phase vs Orientation ','fontsize',15)
    % plot([-phase_cutoff -phase_cutoff],[0 360],'--k');
    % plot([phase_cutoff phase_cutoff],[0 360],'--k');
    % ylim([0 360])
    % xlim([-60 60])
    % plot([min(phase) max(phase) ],[phi_mean phi_mean],'--k');
    
    % coherence and psi
    figure;hold on
    plot(vec_C,phi_temp,'o','markerfacecolor','b');
    %plot(vec_C(SNR_Z>SNR_cutoff),phi_temp(SNR_Z>SNR_cutoff),'o','markerfacecolor','b');
    xlabel('coherence','fontsize',15);ylabel('Orientation angle \psi(\circ)','fontsize',15);
    title('Coherence between C_{rz} and H_{zz} vs Orientation ','fontsize',15)
    plot([0 1],[phi_mean phi_mean],'--k','linewidth',1);
    plot([C_cutoff C_cutoff],[0 360],'--k');
    set(gca,'fontsize',18)
    
    % SNR and psi
    
    figure;hold on;
    
    %plot([SNR_;SNR_Z],[phi_temp;phi_temp],'-o','markerfacecolor','b');
    %plot(SNR_Z(abs(phase)<phase_cutoff),phi_temp(abs(phase)<phase_cutoff),'bo','markerfacecolor','b');
    %plot(SNR_Z,phi_temp,'bo','markerfacecolor','b');
    plot(SNR_R,phi_temp,'bo','markerfacecolor','b');
    %plot(SNR_Z,phi_temp,'ro','markerfacecolor','r');
    
    plot([0 max(SNR_R)],[phi_mean phi_mean],'--k','linewidth',1);
    plot([SNR_cutoff SNR_cutoff],[0 360],'--k');
    
    text(max(SNR_R),phi_mean-10,'\psi=\psi_{mean}');
    %legend('SNR-R','SNR-Z')
    xlabel('SNR','fontsize',15);ylabel('orientation \psi /degrees','fontsize',15);
    title('SNR vs Orientation','fontsize',15)
    
end


%------------plot coverage map-----------------------
if(plot_coverage)
    
    
   % latlim=[-22 -18]; lonlim=[-178 -174];
     latlim=[min([ray(:,1);ray(:,3)]-1) max([ray(:,1);ray(:,3)])+1]; 
     lonlim=[min([ray(:,2);ray(:,4)]-1) max([ray(:,2);ray(:,4)])+1];
    figure;hold on
    h=worldmap(latlim,lonlim);
    %	h=worldmap('world');
    
    
    
    for ip=1:length(vpair)
        h1=plotm([ray(ip,1) ray(ip,3)],[ray(ip,2) ray(ip,4)],'-ko','linewidth',1,'markerfacecolor','k');
    end
    set(h1,'markerfacecolor','k')
    for ip=1:length(ind_pair)
        h2=plotm([ray(ind_pair(ip),1) ray(ind_pair(ip),3)],[ray(ind_pair(ip),2) ray(ind_pair(ip),4)],'-ro','linewidth',1,'markerfacecolor','r');
    end
    
    
    l1=legend([h1,h2],'All inter-station raypath','Used raypath');
    set(l1,'fontsize',15);
    textm(lat1-0.2,lon1-0.7,sta1,'fontsize',15,'color','k');
    geoshow(lat1,lon1,'color','b','marker','^','markersize',12,'markerfacecolor','b');
    
    axesm(gcm,'fontsize',15);
    axesm(gcm,'fedgecolor','none');
end

if(ifsave)
    save orientation.mat sta1 vpair ray phi_temp phi phi_mean std_mean SNR_R SNR_Z SNR_cutoff C_cutoff vec_C
end


cd(dir0)
    % 	end
