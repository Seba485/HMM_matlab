function [raw_probability] = gaussian_classifier(settings, data_set)
%[raw_probability] = gaussian_classifier(settings, data_set)
%settings: structure inside the classifier trained with eegc3_smr_train
%data_set: sample x features
%raw_probability: output of the classifier

m = settings.bci.smr.gau.M;
c = settings.bci.smr.gau.C;

n_class = size(m,1);
n_mixture_per_class = size(m,2);
n_comp = size(m,3);

n_sample = size(data_set,1);



raw_probability = [];

for k = 1:n_sample
    sample = data_set(k,:);
    sample_norm = sample;%./sum(abs(sample));

    likelihood = []; 
    for i = 1:n_class
        for j = 1:n_mixture_per_class
            M = reshape(m(i,j,:),[1 n_comp]);
            C = reshape(c(i,j,:),[1 n_comp]);
            
            determinant = sqrt(C);
            determinant(find(determinant==0))=1;
            determinant = prod(determinant);

            distance = sum(((sample_norm-M).^2)./C);

            likelihood(i,j) = exp(-distance/2) / determinant;
        end
    end
    
    likelihood_sum = sum(likelihood,2)';
    likelihood_norm = likelihood_sum./sum(likelihood_sum);
    raw_probability = [raw_probability; likelihood_norm];
end

            

