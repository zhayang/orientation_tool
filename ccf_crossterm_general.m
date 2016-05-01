% Calculate OBS ambient noise cross correlation record from multiple stationpairs
% for use by calc_orientation.f
% input: daily SAC files of Z,1,2 3C components data from at least 2 OBS.
% 		 locations of both stations
% output : 	for each station in the list calculate Rzz, R1z and R2z, put in dir: ccf/sta0/sta0_stai
%
% 		% Procedures:
% 		1. setup parameters, load staionlist
% 		2. loop through center station
% 		3. loop through other stations, some QC
% 		4. calculate 3 CCFs and stack
% 	 	5. save
%
%% ------------------- settings-------------------
clear
%
 path_save=path;
 rmpath([matlabroot '/toolbox/map/map/']);
 
dir0=pwd;
% % Specify the sampling rate of the input data
% dt=1.0;
% % specify the largest station distances in the network
% maxdist=500;
% % calculate the maximum time lag based on maximun station spacings
% maxlag=maxdist/dt;
%
% % Channel name of Z component:
% chz='LHZ';
%
% % THERE ARE TWO STATION LISTS, THE FIRST ONE IS "CENTERSTATIONLIST":
% % STATIONS TO CALCULATE ORIENTATION FOR
% % THE SECOND ONE IS "STATIONLIST": STATIONS TO SERVE AS VIRTUAL SOURCES
% % WE LOOP THROUGH BOTH
%
% centerstalist= '/Volumes/Work/Lau_research/orientation/ccf/stationlist_whoi.dat'
% % stations to calculate ccf with each center station as virtural sources
% %stationlist = '/Volumes/Work/Lau_research/orientation/ccf/stationlist_whoi.dat'
%
% %
% stationlist = '/Volumes/Work/Lau_research/orientation/ccf/stationlist_all.dat'
% %stationlist= '/Volumes/Work/Lau_research/compliance/test_data/test.dat';
% %stationlist = '/Volumes/Work/Lau_research/orientation/ccf/twostation.dat'
%
% v_sta0=textread(centerstalist,'%s');
% v_sta=textread(stationlist,'%s');
% % sta0 coord
%
% %datadir='/Volumes/Work/Lau_research/OBS_data/';
% % directory of SAC data
% datadir='/Volumes/LauData_YangZha/WHO-antelope/WHO-STAID/';
% %datadir='/Volumes/Work/Lau_research/compliance/test_data/'
% % root directory of target
% ccfdir='/Volumes/Work/Lau_research/orientation/tool/ccf/';
%
% idaystart=1;
% nday=1;
%
% dist_min=60; % minimum distance to calculate CCF
%
% %sta0='';
% ch1=[chz(1),'H1'];
% ch2=[chz(1),'H2'];
setting_tool;
if(~exist('yearstart'))
	yearstart=year;
	yearend=year;
end
%==============SHOULD NOT NEED TO MODIFY ANYTHING  BELOW================
%% ------------------- loop through center station station-------------------


nsta1=length(v_sta0); % number of target stations to calculate for
%nsta1=1;
for ista1=1:nsta1
    sta1=v_sta0{ista1};
    
    %[name1,lat1,lon1,dep1]=textread([datadir,sta1,'/coord.dat'],'%s %f %f %f');% station coordinate
    
    mkdir([ccfdir,sta1]);
    cd([ccfdir,sta1]);
    fpair=fopen('stationpair.txt','w')
    fprintf(fpair,'station  pair    lat1     lon1        dep1        lat2      lon2        dep2       distance      azimuth     back azimuth \n');
    fclose(fpair);
    % loop through stations
    
    % 	create a new pair list,
    % 	comment out if want to append more station pairs to existing list
    %	fpair=fopen('stationpair.txt','w');
    %	fprintf(fpair,'Stations distance azimuth back azimuth  \n');
    %	fclose(fpair);
    
    for ista2=1:length(v_sta)
        clear lat1 lat2 lon1 lon2 dist az baz
        % -------------------read station coord -------------------
        % sta2
        sta2=v_sta{ista2};
        % if same station, skip
        if(strcmp(sta1,sta2))
            continue
        end
        
        if(exist([sta1,'_',sta2,'/Xcorr_',chz,chz,',dat']))
            display('CCF already exist, skip this pair');
            continue
        end
        
        %[name2,lat2,lon2,dep2]=textread([datadir,sta2,'/coord.dat'],'%s %f %f %f');% station coordinate
        
        
        
        % INITIATE STACK
        Rzz_stack=zeros(maxlag*2+1,1);
        R1z_stack=zeros(maxlag*2+1,1);
        R2z_stack=zeros(maxlag*2+1,1);
        SRzz_stack=zeros(maxlag*2+1,1);
        SR1z_stack=zeros(maxlag*2+1,1);
        SR2z_stack=zeros(maxlag*2+1,1);
        
        display(['performing cross-correlation for staion pair : ',sta1,'  ', sta2]);
        % -------------loop through each day--------------------
        nday_stack=0;
        flag_dist=0;
        for year=yearstart:yearend
            if(yearstart==yearend)
                daystart=jdaystart
                dayend=jdayend            
            elseif(year==yearstart)
                daystart=jdaystart
                dayend=366
            elseif(year==yearend)
                daystart=1
                dayend=jdayend
            else
                daystart=1
                dayend=366
            end
            for iday=daystart:dayend
                fclose('all');
                clear Z1 H1 H2 Z2 vec_t R_zz R_1z R_2z SR_zz SR_1z SR_2z
                
                display(['day: ',num2str(iday)]);
                
                if iday<10
                    s = strcat('00',num2str(iday));
                elseif (iday>=10)&&(iday<100)
                    s = strcat('0',num2str(iday));
                elseif (iday>=100)&&(iday<1000)
                    s =num2str(iday);
                end
                syear=num2str(year);
                %------------------- read DATA from data dir------------------------
                datadir1=[datadir,sta1,'/',syear,'/',s,'/'];
                datadir2=[datadir,sta2,'/',syear,'/',s,'/'];
                
                
                %------------------- TEST IF DATA EXIST------------------------
                
                if ~(exist(datadir1)*exist(datadir2))
                    display('no data dir!')
                    continue
                end
                
                
                
                % --------------------read and cut--------------------
                tcut=[0:dt:86000]'; % new time axis to cut to
                filename_z1=[datadir1,'*',chz,'*.SAC*'];
				filename_h1=[datadir1,'*',ch1,'*.SAC*'];
                filename_h2=[datadir1,'*',ch2,'*.SAC*'];
                filename_z2=[datadir2,'*',chz,'*.SAC*'];

				Sz1=readsac(filename_z1); % Z data for station 1
				Sh1=readsac(filename_h1); % H1 data for station 1
				Sh2=readsac(filename_h2); % H2 data for station 1
				Sz2=readsac(filename_z2); % Z data for station 2
				
				if(length(Sz1)*length(Sh1)*length(Sh2)*length(Sz2)~=1)
                    display(['no data for ! station ',sta2]);
                    continue
                end


				% time shift relative to reference 
				tshift_z1 = Sz1.NZHOUR*3600+Sz1.NZMIN*60+Sz1.NZSEC+Sz1.NZMSEC*0.001;
                tshift_h1 = Sh1.NZHOUR*3600+Sh1.NZMIN*60+Sh1.NZSEC+Sh1.NZMSEC*0.001;
                tshift_h2 = Sh2.NZHOUR*3600+Sh2.NZMIN*60+Sh2.NZSEC+Sh2.NZMSEC*0.001;
                tshift_z2 = Sz2.NZHOUR*3600+Sz2.NZMIN*60+Sz2.NZSEC+Sz2.NZMSEC*0.001;

				
%				[vec_tz,Z1raw]=readsac([datadir1,'*',chz,'*.SAC']);
%                [vec_t1,H1raw]=readsac([datadir1,'*',ch1,'*.SAC']);
%                [vec_t2,H2raw]=readsac([datadir1,'*',ch2,'*.SAC']);
%                [vec_tz2,Z2raw]=readsac([datadir2,'*',chz,'*.SAC']);
                
%                 
%                 CALCULATE TIME AXIS
                 vec_tz=tshift_z1+Sz1.B+[0:Sz1.NPTS-1]'*Sz1.DELTA;
                 vec_t1=tshift_h1+Sh1.B+[0:Sh1.NPTS-1]'*Sh1.DELTA;
				 vec_t2=tshift_h2+Sh2.B+[0:Sh2.NPTS-1]'*Sh2.DELTA;
                 vec_tz2=tshift_z2+Sz2.B+[0:Sz2.NPTS-1]'*Sz2.DELTA;
%				 assign DATA 
                 Z1raw=Sz1.DATA1;
				 H1raw=Sh1.DATA1;
				 H2raw=Sh2.DATA1;
                 Z2raw=Sz2.DATA1;

                
                
                s0=size(vec_tz);
                s1=size(vec_t1);
                s2=size(vec_t2);
                s3=size(vec_tz2);
                
                if(s0(1)>10)*(s1(1)>10)*(s2(1)>10)*(s3(1)>10)~=1
                    % if two or more data file exist, skip this day
                    display(['bad data for ! station ',sta2]);
                    continue
                end
                
                if(~exist('lat2','var'));
                    S1=readsac([datadir1,'*',chz,'*.SAC']);
                    S2=readsac([datadir2,'*',chz,'*.SAC']);
                    
                    lat1=S1.STLA;
                    lon1=S1.STLO;
                    dep1=S1.STEL; % depth is negative for OBS and positive for land stations
                    
                    
                    lat2=S2.STLA;
                    lon2=S2.STLO;
                    dep2=S2.STEL; % depth is negative for OBS and positive for land stations
                    
                    
                    % correct for LDEO data longitude header which is wrong
                    %                 if(sta1(4)=='L');
                    %                     lon1=-lon1;
                    %                 end
                    %                 if(sta2(4)=='L');
                    %                     lon2=-lon2;
                    %                 end
                    %
                    
                    % use home made distance code independent from mapping
                    % toolbox
                    
                    [delta,az]=coord2dist(lat1,lon1,lat2,lon2);
                    [delta,baz]=coord2dist(lat2,lon2,lat1,lon1);
                    
%                       [delta,az]=distance(lat1,lon1,lat2,lon2);
%                     [delta,baz]=distance(lat2,lon2,lat1,lon1);
                    dist=delta*6371*pi/180;
                    
                    Delta=S1.DELTA;
                    if(abs(Delta-dt)>=1e-6)
                        error('sampling interval does not match data! check dt');
                    end
%                     dist
                    if(dist<dist_min)
                        display('distance shorter than 60 km, skip');
                        flag_dist=1;
                        break
                    end
                end
                
                Z1=interp1(vec_tz,Z1raw,tcut);
                Z1(isnan(Z1))=0;
                
                
                H1=interp1(vec_t1,H1raw,tcut);
                H1(isnan(H1))=0;
                
                
                H2=interp1(vec_t2,H2raw,tcut);
                H2(isnan(H2))=0;
                
                % Z2
                
                Z2=interp1(vec_tz2,Z2raw,tcut);
                Z2(isnan(Z2))=0;
                
                
                % detrend
                Z1=detrend(Z1);
                Z2=detrend(Z2);
                H1=detrend(H1);
                H2=detrend(H2);
                
                
                % %----------calculate daily CCF and stack%-------------------
                R_zz = xcorr(Z1,Z2,maxlag);
                R_1z = xcorr(H1,Z2,maxlag);
                R_2z = xcorr(H2,Z2,maxlag);
                
                % make CCF symmetric
                SR_zz = (R_zz+flipud(R_zz))/2;
                SR_1z = (R_1z+flipud(R_1z))/2;
                SR_2z = (R_2z+flipud(R_2z))/2;
                % STACK daily normalized CCF
                
                % skip NANS
                % DATA QC
                if(sum(isnan(R_zz))~=0)||(sum(R_zz~=0)==0)
                    display('skip');
                    continue
                end
                if(sum(isnan(R_1z))~=0)||(sum(R_1z~=0)==0)
                    
                    display('skip');
                    continue
                end
                if(sum(isnan(R_2z))~=0)||(sum(R_2z~=0)==0)
                    
                    display('skip');
                    continue
                end
                % normalize for each day, normalize Rzz, R1z and R2z same way to preserve relative amplitude
                Rzz_stack=Rzz_stack+R_zz/max(abs(R_zz));
                R1z_stack=R1z_stack+R_1z/(max(abs(R_1z))+max(abs(R_2z)))*2;
                R2z_stack=R2z_stack+R_2z/(max(abs(R_1z))+max(abs(R_2z)))*2;
                
                SRzz_stack=SRzz_stack+SR_zz/max(abs(R_zz));
                SR1z_stack=SR1z_stack+SR_1z/(max(abs(R_1z))+max(abs(R_2z)))*2;
                SR2z_stack=SR2z_stack+SR_2z/(max(abs(R_1z))+max(abs(R_2z)))*2;
                
                nday_stack=nday_stack+1;
                
            end
            
            if(flag_dist==1)
                break
            end
        end
        % finish and save
        
        tlag=[-maxlag:maxlag]*dt;
        Xcorr_zz=[tlag(:) SRzz_stack(:) Rzz_stack(:)];
        Xcorr_1z=[tlag(:) SR1z_stack(:) R1z_stack(:)];
        Xcorr_2z=[tlag(:) SR2z_stack(:) R2z_stack(:)];
        
        if(exist('dist','var'))
            filenamez=['Xcorr_',chz,chz,'.dat'];
            filename1=['Xcorr_',ch1,chz,'.dat'];
            filename2=['Xcorr_',ch2,chz,'.dat'];
            %		pairdir=[ccfdir,sta1,'/',sta1,'_',sta2];
            
            cd([ccfdir,sta1])
            mkdir([sta1,'_',sta2]);
            cd([sta1,'_',sta2]);
            
            save(filenamez,'Xcorr_zz','-ascii')
            save(filename1,'Xcorr_1z','-ascii')
            save(filename2,'Xcorr_2z','-ascii')
            
            % write distance etc to file
            fmt = '%5f %5f %5f\n';
            
            fid = fopen('distance.dat','w');
            fprintf(fid,'distance   azimuth   back azimuth \n');
            fprintf(fid, fmt, dist,az,baz);
            fclose(fid);
            
            fid = fopen('info.txt','w');
            fprintf(fid,'Days stacked \n');
            fprintf(fid, '%d \n',nday_stack);
            fclose(fid);
            
            
            cd([ccfdir,sta1]);
            fpair=fopen('stationpair.txt','a');
            fprintf(fpair,'%s  %5f   %5f  %5f  %5f  %5f  %5f   %5f   %5f   %5f  \n',[sta1,'_',sta2],lat1,lon1,dep1,lat2,lon2,dep2,dist,az,baz);
            fclose(fpair);
            
            
        end
        
    end
    
    
    
    
end
path(path_save)
cd(dir0)


