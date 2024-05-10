function [samples] = sample_generator_gaussian(mu, sigma, weigth, n_sample)
% [samples] = sample_generator_gaussian(mu, sigma, weigth, n_sample)
% this function will create the samples for the distribution with the
% specified parameters
% INPUT: mu --> means of the distribution
%        sigma --> standard deviation
%        weigth --> weight for each gaussian (if it is only one it has to be 1)
%        n_sample --> number of sample fo the distribution

%each class has to be within 0-1 and has to have the same number of 
%samples, in order to esure that an higer number of data will be created 
%and then a cut will be performed.
    
    aug_n_sample = round(3*n_sample);

    samples = [];
    for k = 1:length(weigth)
        samples = [samples; normrnd(mu(k),sigma(k),[round(weigth(k)*aug_n_sample),1])];
    end
    %shuffle all the sample
    samples(samples<0|samples>1) = [];
    samples = samples(randperm(length(samples)));
    samples = samples(1:n_sample);
end