function [trial_classification, trial_accuracy] = Trial_accuracy(y,trial_flag,true_label,th,CODE)
%[trial_classification, trial_accuracy] = Trial_accuracy(y,trial_flag,true_label,th,CODE)
%Given the control framework the function computes the trial classifcation and accuracy 
%y --> controll framework output
%trial_flag --> indexes of the trials start
%truel_label --> true label for each data
%CODE --> struct with the experimental codes
%OUTPUT: trial_classification --> classificatiokn for each trial
%        trial_accuracy --> classification accuracy at trial level

    Ck_label = [];
    for k = 1:length(y)
        if y(k)>=th(2)
            Ck_label(k) = CODE.Both_Feet;
        end
        if y(k)<=th(1)
            Ck_label(k) = CODE.Both_Hand;
        end
        if y(k)>=th(1) && y(k)<=th(2)
            Ck_label(k) = CODE.Rest;
        end 
    end
    
    trial_classification = [];
    for k = 1:length(trial_flag)
        if k == length(trial_flag)
            rest_count = length(find(Ck_label(trial_flag(k):end)==CODE.Rest));
            hand_count = length(find(Ck_label(trial_flag(k):end)==CODE.Both_Hand));
            feet_count = length(find(Ck_label(trial_flag(k):end)==CODE.Both_Feet));
        else
            rest_count = length(find(Ck_label(trial_flag(k):trial_flag(k+1))==CODE.Rest));
            hand_count = length(find(Ck_label(trial_flag(k):trial_flag(k+1))==CODE.Both_Hand));
            feet_count = length(find(Ck_label(trial_flag(k):trial_flag(k+1))==CODE.Both_Feet));
        end

        task_codes = [CODE.Rest CODE.Both_Hand, CODE.Both_Feet];
        task_counts = [rest_count hand_count feet_count];
        max_idx = find(task_counts==max(task_counts)); %in case of equal counts take the first one
        trial_classification(k) = task_codes(max_idx(1));
    end
    
    true_trial_label = true_label(trial_flag);
    
    trial_accuracy = sum(trial_classification'==true_trial_label)/length(trial_classification);
end