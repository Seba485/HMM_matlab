function [selected_ch,selected_freq,FS_matrix,PSD_reshape,Ck_win] = Fisher_matrix(PSD_signal,h_PSD,CODE,th)
%[selected_ch,selected_freq,FS_matrix,PSD_reshape] = Fisher_matrix(PSD_signal,h_PSD,CODE)
%PSD_signal --> PSD of the signal [windows x frequency x channels]
%h_PSD --> struct with the information related to the PSD
%CODE --> struct with the exprimental codes
%th --> threshold for feature selection
%OUTPUT: Ck_win --> label for each window of interest
%        PSD_reshape --> [windows(CUE-->Continuous feedback) x frequency*channels]
%        FS_matrix --> Fisher matrix
%        selected_freq and selected_ch --> feature selected throw a threshold applyed to the Fisher matrix

    ch = size(PSD_signal,3);
    freq = size(PSD_signal,2);
    % idexes of interest CUE-->Continuous_feedback
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
    
    selected_features = sub2ind(size(FS_matrix'),selected_freq,selected_ch);
end
