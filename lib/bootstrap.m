%bootstrap method to estimate circular mean and confidence level of  distribution 
function [v_mean mu_est mu_std] = bootstrap(data,Nr)
% This function use bootstrap method (resampleing) to estimate the mean its variance of a un-parameterized sample
% input: data: sample data (in degrees)
% 		 Nr: number of times to resample
% output: v_mean: 1 by N vector of mean for each resampling realization
% 			mu_est: mean of the mean
% 			mu_std: standard deviation of the mean using linear estimates
%
	n=length(data);
	for ir=1:Nr
		ind_r = randi(n,[1 n]); % generate a vector of n random index from 1-n used for resample;
		rdata = data(ind_r);
		v_mean(ir)=mean(rdata);			
	end
	mu_est = mean(v_mean);
	mu_std = std(v_mean);

	
end	
	

