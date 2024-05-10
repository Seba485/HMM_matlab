function [s,t,h,count] = Artifact_Trial_removal(s,t,h,ch_ref,limit,CODE)
%[s,t,h,count] = Artifact_Trial_removal(s,t,h,ch_ref,limit,CODE)
%Removal of the trials currapted by artifacts
%s --> signal
%t --> time base
%h --> structure related to the signal
%ch_ref --> reference channel for the artifact detection 
%limit --> uV of absolut amplitude, threshold for the artifact detection
%CODE --> structure that contain the codes of the trial parts
%OUTPUT: updated s, t and h, count-->number of removed trial 


    trial_start_idx = find(h.EVENT.TYP==CODE.Trial_start);
    trial_start_pos = h.EVENT.POS(trial_start_idx);
    n_trial = length(trial_start_pos);
    s_idx = [];
    h_idx = [];
    count = 0;
    for k = 1:n_trial
        if k==n_trial
            trial_pos = trial_start_pos(k):length(t);
            trial_idx = trial_start_idx(k):length(h.EVENT.TYP);
        else
            trial_pos = trial_start_pos(k):trial_start_pos(k+1)-1;
            trial_idx = trial_start_idx(k):trial_start_idx(k+1)-1;
        end
        signal = s(trial_pos,ch_ref);
        artifact_pos = find(signal>=limit|signal<=-limit);
        if isempty(artifact_pos)
            %no artifact detected
        else
            %trial rejection
            count = count+1;
            s_idx = [s_idx, trial_pos];
            h_idx = [h_idx, trial_idx];
            pos_shift = length(trial_pos);
            if k<n_trial
                h.EVENT.POS(trial_idx(end)+1:end) = h.EVENT.POS(trial_idx(end)+1:end)-pos_shift; 
            end
            h.EVENT.POS(h_idx) = 0;
        end
    end
    s(s_idx,:) = [];
    t(s_idx) = [];
    h.EVENT.DUR(h_idx) = [];
    h.EVENT.TYP(h_idx) = [];
    h.EVENT.POS(h_idx) = [];
    h.N = length(t);

end