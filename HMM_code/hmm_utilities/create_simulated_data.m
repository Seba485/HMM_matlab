function [trn_pp, true_label, trial] = create_simulated_data(CODE,n,t,f)
% [trn_pp, Ck_trn] = create_simulated_data(CODE,n,t,f,'type','gaussian')
% create the simulated data for the experiment,
% 3 distribution, one near zero for the both_feet class, one near 1 for the
% both_hand class and one bimodal for the rest_class
% INPUT: CODE --> structure that contain the codes for each task
%        n --> number of total sample to generate
%        t --> duration in seconds of the simulated tasks
%        f --> output frequency
% OUTPUT: trn_pp --> simulated probability output
%         Ck_trn --> label for each sample
% Addictional info: to emulate the real condition each task will last 10 
% seconds and the series of task wil be casual, in order to ensure a large
% number of sample for each class select a high n (>10000)
    
    trn_pp = [];
    true_label = [];

    trial.start = [];
    trial.label = [];

    label = [CODE.Both_Hand, CODE.Both_Feet, CODE.Rest];
    n_task = length(label);

    n_sample = round(t*f); %number of sample per task

    added_sample = 0;
    while added_sample < n
        random_task = randi([1 n_task]);
        trial.label = [trial.label; label(random_task)];
        trial.start = [trial.start;1;zeros(n_sample-1,1)];
    
        true_label = [true_label; label(random_task)*ones(n_sample,1)];

        switch random_task
            case 1 %both hand
                samples = sample_generator_gaussian([0 0.1], [0.05 0.1], [0.9 0.1], n_sample);
            case 2 %both feet
                samples = sample_generator_gaussian([1 0.9], [0.05 0.1], [0.9 0.1], n_sample);
            case 3 %rest
                samples = sample_generator_gaussian([0.1 0.9], [0.2 0.2], [0.5 0.5], n_sample);
            otherwise
                %pass
        end

        trn_pp = [trn_pp; samples];

        added_sample = length(trn_pp);
    end

end
        


