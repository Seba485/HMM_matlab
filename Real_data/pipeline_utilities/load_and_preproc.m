function [data, trial] = load_and_preproc(settings, folder_path, type, CODE)
% [data, trial] = load_and_preproces(settings, folder_path, type, CODE)
% settings: structure inside the .smr.mat binary classifier
% folder_path: complete path in which there are the 3 classes recordings
% type: 'calibration' 'evaluation'
% CODE: structure with the codes of the task
% data: structure: data.data: sample x features (only sample from cue-->end continuous feedback)
%                  data.label: label for sample
%                  data.n_sample: number of sample
%                  data.f: frequency of output
%trial: structure: trial.POS: concat of h.EVENT.POS of the files
%                  trial.TYP: concat of h.EVENT.TYP of the files
%                  trial.DUR: concat of h.EVENT.DUR of the files
%                  trial.info: string with info about the trials
%                  trial.start: zero array with 1 in the start instant of each trial
%                  trial.len: array with the length of each trial
%                  trial.lable: array with the lables for trial
%                  trial.idx: indexes of usefull sample (cue-->end of continuous feedback)
%                  trial.n: number of trial

    %% Parameter
    
    % General
    n_ch = settings.acq.channels_eeg;
    ch_names = settings.acq.channel_lbl; %cell array
    
    Fs = settings.acq.sf; %seample frequency
    
    electrode_map = settings.modules.smr.montage; %it could be numerical
    %load('electrode_map_'+string(n_ch)+'.mat')
    
    task = settings.bci.smr.taskset.classes;
    task_name = {settings.bci.smr.taskset.modality(1:2), settings.bci.smr.taskset.modality(3:4)};
    
    %filter
    a = settings.modules.smr.options.prep.filter.a;
    b = settings.modules.smr.options.prep.filter.b;
    
    %PSD
    lap = settings.modules.smr.laplacian; %laplacian filter
    win_size = settings.modules.smr.win.size;
    win_shift =  settings.modules.smr.win.shift;
    internal_win_size = settings.modules.smr.psd.win;
    overlap = settings.modules.smr.psd.ovl;
    pshift = internal_win_size*overlap; % seconds. Shift of the internal windows
    psd_freq = settings.modules.smr.psd.freqs;

    data.f = 1/win_shift;
    
    %features
    ch_feature = settings.bci.smr.channels;
    freq_feature = settings.bci.smr.bands;
    
    %classifier
    m = settings.bci.smr.gau.M; %means of the gmm: classes x component x dimension 
    c = settings.bci.smr.gau.C; %covariance of the gmm: classes x component x dimension 
    
    
    %% load files (3 classes)
    
    root = [folder_path '/'];
    file_info = dir(root);
    file_name = {file_info.name};
    
    s = {}; %cell with the calibration run in the folder
    h = {}; %cell with the header fo each calibration run in the folder
    i = 0;
    for k = 1:length(file_name)
        if isempty(strfind(file_name{k},type))
            %pass
        else
            disp(file_name{k})
            i = i+1;
            [s{i}, h{i}] = sload([root file_name{k}]); %signal and header
            h{i}.N_sample = size(s{1},1);
            h{i}.time_base = [0:h{i}.N_sample-1]./h{1}.SampleRate;
        end
    end
    
    n_run = length(s);

    if n_run==0 %if there's no data
        data = NaN;
        trial = NaN;
        return;
    end
    
    %% Filter data
    
    %band pass filter 1 - 40 Hz since im looking for information in mu and
    %beta band, namely in 8-13Hz and 13-30Hz
    
    s_filt = {};
    for k = 1:n_run
        s_filt{k} = filtfilt(b,a,s{k});
    end
    
    
    %% Laplacian filter
    
    % load('electrode_map_'+string(n_ch)+'.mat')
    % lap = laplacian_filter(n_ch,ch_names,electrode_map);
    
    %% PSD
    h_PSD = {};
    PSD_signal = {};
    
    for k = 1:n_run
    
        %applicazione del filtro 
        s_laplacian = s_filt{k}(:,1:n_ch) * lap;
        
        [PSD, f] = proc_spectrogram(s_laplacian, internal_win_size, win_shift, pshift, Fs, win_size);

        %LOG PSD!!!!!!!!!
        PSD = log(PSD);
        
        %select meaningfull frequences 
        h_PSD{k}.f = f(find(f==psd_freq(1)):find(f==psd_freq(end)));
        PSD_signal{k} = PSD(:,find(f==psd_freq(1)):find(f==psd_freq(end)),:);
        
        %recompute the Position of the event related to the PSD windows
        h_PSD{k}.EVENT.POS = proc_pos2win(h{k}.EVENT.POS, win_shift*Fs, 'backward', internal_win_size*Fs);
        h_PSD{k}.EVENT.DUR = round(h{k}.EVENT.DUR./(win_shift*Fs));
        h_PSD{k}.EVENT.TYP = h{k}.EVENT.TYP;
    end

    %psd -> win x frequency x channels
    
    %% Data set (data_set (sample x features) - true_lable)
    
    %reshape different runs
    
    PSD_reshape = {};
    
    for run = 1:n_run
        PSD_reshape{run} = [];
        n = 0;
        for ch = 1:length(ch_feature)
            freq_vect = freq_feature{ch_feature(ch)};
            for freq = 1:length(freq_vect)
                n = n + 1;
                PSD_reshape{run}(:,n) = PSD_signal{run}(:,find(h_PSD{run}.f==freq_vect(freq)),ch_feature(ch));
            end
        end 
    end
    
    %merge into a uniqeu dataset
    
    data.data = [];
    data.label = [];
    
    trial.POS = [0];
    trial.TYP = [];
    trial.DUR = [];
    
    for run = 1:n_run
        data.data = [data.data; PSD_reshape{run}];
    
        trial.POS = [trial.POS; trial.POS(end)+h_PSD{run}.EVENT.POS]; 
        trial.TYP = [trial.TYP; h_PSD{run}.EVENT.TYP];
        trial.DUR = [trial.DUR; h_PSD{run}.EVENT.DUR];
    end
    trial.POS(1) = [];
    
    % the training set is made up all the cue-->end continuous feedback period
    % in order to have the information about trials we need to manipulate the
    % struct trial
    trial.info = 'trial start and label refers to cue-->end of continuous feedback period';
    trial.start = [];
    
    trial_start = trial.POS(trial.TYP==task(1) | trial.TYP==task(2) | trial.TYP==CODE.Rest); %cue
    trial_end = trial.POS(trial.TYP==CODE.Continuous_feedback) + trial.DUR(trial.TYP==CODE.Continuous_feedback); %end of continuous feedback
    trial.len = trial_end - trial_start; %length of each trial
    
    n_trial = length(trial.len);
    trial.label = zeros(n_trial,1);
    trial.idx = []; %indexes of the actual trials
    for k = 1:n_trial
        trial.start = [trial.start; 1; zeros(trial.len(k)-1,1)];
        trial.label(k) = trial.TYP(find(trial.POS==trial_start(k)));
    
        trial.idx = [trial.idx; (trial_start(k):trial_end(k)-1)'];
        data.label = [data.label; repelem(trial.label(k),trial.len(k))'];
    end
    trial.n = length(trial.label);
    
    % maintain only data related to the trials
    data.data = data.data(trial.idx,:);
    data.n_sample = length(data.data);

end


