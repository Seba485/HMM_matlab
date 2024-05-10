function [s,t,h] = LoadAndConcatenate_data(dataset_path,subject_folder,type)
%[s,t,h] = LoadAndConcatenate_data(dataset_path,subject_folder,type)
%dataset_path --> path of the forlder that contains the subjects folders
%subject_folder --> name of the subject folder
%type --> 'online' or 'offline'
%RETURN: s --> concatenated offline signal
%        t --> time base
%        h --> struct that contains: Fc --> sample frequency
%                                    ch_names --> name of channels (in order)
%                                    N --> number of sample
%                                    EVENT --> struct that contains TYP,
%                                              DUR, POS related to the
%                                              concatenated signal

    root = [dataset_path '/' subject_folder '/'];
    file_info = dir(root);
    file_name = {file_info.name};
    
    s = [];
    t = [0];
    h.EVENT.POS = [];
    h.EVENT.TYP = [];
    h.EVENT.DUR = [];
    
    for k = 1:length(file_name)
        if isempty(strfind(file_name{k},type))
            %pass
        else
            [s_, h_] = sload([root file_name{k}]); %signal and header
            
            Fs = h_.SampleRate;
            N_ = size(s_,1);
            t_ = [0:1/Fs:(N_-1)/Fs]';
            
            %signal
            pos_bias = length(s);
            s = [s; s_];
            t = [t; t_+t(end)];
    
            %events
            h.EVENT.POS = [h.EVENT.POS; h_.EVENT.POS+pos_bias];
            h.EVENT.TYP = [h.EVENT.TYP; h_.EVENT.TYP];
            h.EVENT.DUR = [h.EVENT.DUR; h_.EVENT.DUR];
    
        end
    end
    t = t(2:end);
    
    h.N = size(s,1);
    h.ch_names = ["Fz","FC3","FC1","FCz","FC2","FC4","C3","C1","Cz","C2","C4","CP3","CP1","CPz","CP2","CP4","Ref"];
    h.SampleRate = Fs;


end

