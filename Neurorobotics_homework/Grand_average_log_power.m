function [Log_avg] = Grand_average_log_power(s,h,f_band,trial_span,task_codes)
%[Log_avg] = Grand_average_log_power(s,h,f_band,trial_span, task_codes)
%Log_avg --> average log power in the f_band for all the trial type
%s --> concatenated signal
%h --> struct with the information about the signal
%f_band --> vector of lenth 2: [low_freq high_freq]
%trial_span --> [inital_event final_event] of a trial
%task_codes --> vector with the code of the tasks


    n = 5; %filter order
    span = 1; %s window of the moving average
    
    [b,a] = butter(n,f_band./(h.SampleRate/2),"bandpass");
    
    sign_filt = filtfilt(b,a,s);
    
    % rectifing the signal
    sign_rect = abs(sign_filt).^2;
    
    %moving average
    a = 1;
    windows_len = h.SampleRate*span;
    b = ones(1,windows_len)./windows_len;
    
    sign_avg = filtfilt(b,a,sign_rect);
    
    %log trasform
    sign_log = log(sign_avg);
    
    %Trial: fixation cross - Continuous feedback period [sample x channels x trial]
    [log_trial_matrix,Ck] = Signal_into_trial(sign_log,h,trial_span,task_codes);
    
    
    %Grand averaging Log power
    
    Log_avg = [];
    for k = 1:length(task_codes)
        Log_avg(:,:,k) =  mean(log_trial_matrix(:,:,Ck==task_codes(k)),3);
    end
end