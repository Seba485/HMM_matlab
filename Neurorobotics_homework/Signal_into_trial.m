function [trial_matrix,Ck] = Signal_into_trial(s,h,trial_span,task_codes)
%[trial_matrix,Ck] = Signal_into_trial(s,h,trial_span,task_codes)
%trial_matrix --> signal divided into trial: 3D matrix [sample x channels x trial]
%Ck --> vector vith the label for each trial
%s --> signal
%h --> struct with the information related to the signal
%trial_span --> [initial_code end_code]
%task_codes --> code of the tasks to label

    %split the signal per trial
    start_pos = find(h.EVENT.TYP==trial_span(1));
    end_pos = find(h.EVENT.TYP==trial_span(2));
    
    N_trial = sum(h.EVENT.TYP==trial_span(1));
    
    trial_len = [];
    for k = 1:N_trial
        start = h.EVENT.POS(start_pos(k));
        stop = h.EVENT.POS(end_pos(k))+h.EVENT.DUR(end_pos(k))-1;
        trial_len(k) = stop-start;
    end
    
    trial_len = min(trial_len); %is necessary to take a precise length for all the trial
    
    trial_matrix = [];
    
    if size(s,3)>1
        for k = 1:N_trial
            start = h.EVENT.POS(start_pos(k));
            stop = start+trial_len-1;
            trial_matrix(:,:,:,k) = s(start:stop,:,:);
        end
    else
        for k = 1:N_trial
            start = h.EVENT.POS(start_pos(k));
            stop = start+trial_len-1;
            trial_matrix(:,:,k) = s(start:stop,:);
        end
    end
    
    %label per trial
    
    Ck = [];
    for k = 1:length(h.EVENT.TYP)
        if ismember(h.EVENT.TYP(k),task_codes)
            Ck = [Ck; h.EVENT.TYP(k)];
        end
    end
end
