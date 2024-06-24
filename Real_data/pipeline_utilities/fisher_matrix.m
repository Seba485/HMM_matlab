function [feature_table,reshape_idx,FS_matrix,PSD_reshape,Ck_win] = fisher_matrix(th,PSD_signal,h_PSD,CODE,ch_names)
%[ch_f_feature_idx,selected_features_idx,FS_matrix,PSD_reshape] = Fisher_matrix(th,PSD_signal,h_PSD,CODE,ch_names)
%PSD_signal --> PSD of the signal [windows x frequency x channels]
%h_PSD --> struct with the information related to the PSD
%CODE --> struct with the exprimental codes
%th --> threshold for feature selection
%ch_names --> string array with the channel name in the rigth order
%OUTPUT: Ck_win --> label for each window of interest
%        PSD_reshape --> [windows(CUE-->Continuous feedback) x frequency*channels]
%        FS_matrix --> Fisher matrix
%        reshape_idx --> index of the selected features for the PSD_reshaped matrix
%        ch_f_feature_idx --> first column: index of the channel, second colums: index of the frequency
   

    ch = size(PSD_signal,3);
    freq = size(PSD_signal,2);
    % idexes of interest CUE-->Continuous_feedback for al the trial 
    start_pos = find(h_PSD.EVENT.TYP==CODE.Both_Hand | h_PSD.EVENT.TYP==CODE.Both_Feet | h_PSD.EVENT.TYP==CODE.Rest);
    end_pos = find(h_PSD.EVENT.TYP==CODE.Continuous_feedback);
    idx = [];
    Ck_win = []; %label of idx
    for k = 1:length(start_pos)
        idx_trial = [h_PSD.EVENT.POS(start_pos(k)):(h_PSD.EVENT.POS(end_pos(k))+h_PSD.EVENT.DUR(end_pos(k))-1)]';
        idx = [idx; idx_trial];

        Ck_win = [Ck_win; repelem(h_PSD.EVENT.TYP(start_pos(k)),length(idx_trial))'];

    end
    
    PSD_reshape = reshape(PSD_signal(idx,:,:),[length(idx),freq*ch]); 
    %windows of interest x freq*ch
    
    % Features Selection
    Activity_Hand = PSD_reshape(Ck_win==CODE.Both_Hand,:,:);
    Activity_Feet = PSD_reshape(Ck_win==CODE.Both_Feet,:,:);
    
    Hand_mean = mean(Activity_Hand,1);
    Feet_mean = mean(Activity_Feet,1);
    Hand_std = std(Activity_Hand);
    Feet_std = std(Activity_Feet);
    
    %fisher computation
    FS = zeros(1,freq*ch);
    for k = 1:(freq*ch)
        FS(k) = abs(Hand_mean(k)-Feet_mean(k))/sqrt(Hand_std(k)^2 + Feet_std(k)^2);  
    end
    
    FS_matrix = reshape(FS,[freq,ch])';
    
    [selected_ch, selected_freq] = find(FS_matrix>=th); %chosen looking at the matrix
    
    reshape_idx = sub2ind(size(FS_matrix'),selected_freq,selected_ch);
    %on the matrix : windows of interest x freq*ch

    ch_f_idx = [selected_ch, selected_freq, h_PSD.f(selected_freq)];
    
    feature_table = array2table([ch_names(ch_f_idx(:,1))', ch_f_idx],"VariableNames",["ch name" "ch idx" "freq idx" "freq Hz"]);
    

end