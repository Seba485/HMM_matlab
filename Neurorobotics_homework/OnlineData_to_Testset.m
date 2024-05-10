function [test_set, framework_testset, tot, trial_flag] = OnlineData_to_Testset(PSD_online,h_PSD_online,selected_features,CODE)
%[test_set, framework_testset, tot, trial_flag] = OnlineData_to_Testset(PSD_online,h_PSD_online,selected_features,CODE)
%PSD_online --> PSD of the concatenated online files
%h_PSD_online --> struct that contain all the information about the concatenated online file
%selected_features --> indexes of the selected features
%OUTPUT: test_set --> [windows(only task) x feature+1] the last column are the labels
%        frame_work_testset --> [windows(task + rest) x features+1] the last column are the labels
%        tot --> struct: test_data --> tot data available on test set
%                        test_hand --> tot task both hand
%                        test_feet --> tot task both feet
%        trial_flag --> vector of inedxes of trial start

    ch = size(PSD_online,3);
    freq = size(PSD_online,2);
    
    % idexes of interest ONLY Continuous_feedback
    start_pos = find(h_PSD_online.EVENT.TYP==CODE.Both_Hand | h_PSD_online.EVENT.TYP==CODE.Both_Feet | h_PSD_online.EVENT.TYP==CODE.Rest); %for the lable
    end_pos = find(h_PSD_online.EVENT.TYP==CODE.Continuous_feedback);
    %in the online datas there is also the rest phase that is not usefull for
    %calculating the accuracy of the model on test set, but its usefull to compute
    %the trial accuracy using the framework.
    idx_test = []; %for the accuracy
    idx = []; %for the graph of the framework
    Ck_win = []; %label of idx
    trial_flag = []; %says when to reset the framework
    for k = 1:length(start_pos)
        idx_trial = [h_PSD_online.EVENT.POS(end_pos(k)):((h_PSD_online.EVENT.POS(end_pos(k))+h_PSD_online.EVENT.DUR(end_pos(k))-1))]';
        idx = [idx; idx_trial];
        if h_PSD_online.EVENT.TYP(start_pos(k)) ~= CODE.Rest
            idx_test = [idx_test; idx_trial];
        end
    
        Ck_win(idx_trial) = h_PSD_online.EVENT.TYP(start_pos(k)); 
        
        trial_flag = [trial_flag; 1; zeros(length(idx_trial)-1,1)];  
    end
    
    trial_flag = find(trial_flag==1);

    PSD_reshape_test = reshape(PSD_online(idx_test,:,:),[length(idx_test),freq*ch]);
    
    PSD_reshape = reshape(PSD_online(idx,:,:),[length(idx),freq*ch]);
    framework_testset = PSD_reshape(:,selected_features);
    framework_testset(:,end+1) = Ck_win(idx);
    
    %creation of the test set
    test_set = PSD_reshape_test(:,selected_features);
    test_set(:,end+1) = Ck_win(idx_test);
    tot.test_data = size(PSD_reshape_test,1);
    tot.test_hand = sum(Ck_win==CODE.Both_Hand);
    tot.test_feet = sum(Ck_win==CODE.Both_Feet);
end
