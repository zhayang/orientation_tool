%bootstrap method to estimate circular mean and confidence level of distribution mean of circular data
%All input and out put are in degrees
function [v_mean mu_est mu_std] = bootstrap_circ(data,Nr)
% This function use bootstrap method (resampleing) to estimate the mean its variance of a un-parameterized circular sample
% input: data: sample data (in degrees)
% 		 Nr: number of times to resample
% output: v_mean: 1 by N vector of mean for each resampling realization
% 			mu_est: mean of the mean
% 			mu_std: standard deviation of the mean using linear estimates
% 			range of v_mean and mu_est: [0 360]
%
	data=data(:);
	n=length(data);
	data_rad=data*pi/180; % convert degree to rad for conveniece 
	for ir=1:Nr
		ind_r = randi(n,[1 n]); % generate a vector of n random index from 1-n used for resample;
		rdata = data_rad(ind_r); % resample
		v_mean_rad(ir)=circ_mean(rdata);			
	end
	size(v_mean_rad);
	v_mean_rad=v_mean_rad(:);
	mu_est = circ_mean(v_mean_rad)*180/pi;
	mu_std = circ_std(v_mean_rad)*180/pi;
	v_mean = v_mean_rad*180/pi;
	
	v_mean=v_mean+360*(v_mean<0);
	mu_est=mu_est+360*(mu_est<0);
	
end	
	


