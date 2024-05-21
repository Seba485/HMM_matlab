clear
close all
clc

CODE.Trial_start = 1;
CODE.Fixation_cross = 786;
CODE.Both_Hand = 773;
CODE.Both_Feet = 771;
CODE.Rest = 783;
CODE.Continuous_feedback = 781;
CODE.Target_hit = 897;
CODE.Target_miss = 898;

%% Load Calibration data
dataset_path = '/home/sebastiano/HMM_matlab/Real_data/load_file/calibration_data';
subject_folder = '20231213_d7';
type = 'calibration';

root = [dataset_path '/' subject_folder '/'];
file_info = dir(root);
file_name = {file_info.name};

% im gonna study the runs separately and then onec the features are chosen
% ill merge the data into a unique data stream
s = {}; %cell with the calibration run in the folder
h = {}; %cell with the header fo each calibration run in the folder
i = 0;
for k = 1:length(file_name)
    if isempty(strfind(file_name{k},type))
        %pass
    else
        i = i+1;
        [s{i}, h{i}] = sload([root file_name{k}]); %signal and header
        h{i}.N_sample = size(s{1},1);
        h{i}.time_base = [0:h{i}.N_sample-1]./h{1}.SampleRate;
    end
end
clear k i file_name file_info root dataset_path type subject_folder

% is the same for all the runs
Fs = h{1}.SampleRate; 
n_ch = size(s{1},2);
n_run = length(s);

ch_names = strings(length(h{1}.Label),1);
for k = 1:length(ch_names)
    ch_names(k) = h{1}.Label{k};
end
load("load_file/electrode_map.mat") %electrode map
%% Filter data

%band pass filter 1 - 40 Hz since im looking for information in mu and
%beta band, namely in 8-13Hz and 13-30
fc = [1 30];
[b_bp,a_bp] = butter(5,fc/(Fs/2),'bandpass');

figure(1)
freqz(b_bp,a_bp,[],Fs)
title('Preliminary signal filter')

s_filt = {};
for k = 1:n_run
    s_filt{k} = filtfilt(b_bp,a_bp,s{k});
end
clear k b_bp a_bp

figure(2)
ch_idx = find(ch_names=='C3' | ch_names=='CZ' | ch_names=='C4');
subplot(211)
plot(h{1}.time_base,s{1}(:,ch_idx))
legend('C3', 'Cz', 'C4')
title('Original signal')
grid on
xlim([h{1}.time_base(1) h{1}.time_base(end)])
xlabel('t[sec]')
ylabel('eeg[\muV]')

subplot(212)
plot(h{1}.time_base,s_filt{1}(:,ch_idx))
legend('C3', 'Cz', 'C4')
title('Filtered signal')
grid on
xlim([h{1}.time_base(1) h{1}.time_base(end)])
ylim([-150 +150])
xlabel('t[sec]')
ylabel('eeg[\muV]')

% power spectra
channel = 'CZ';
run = 1;

power_spectra = abs(fft(s_filt{run}(:,find(ch_names==channel)))).^2;
power_spectra = power_spectra(1:floor(h{1}.N_sample/2));
f_base = [0:h{1}.N_sample-1].*Fs/h{1}.N_sample-1;
f_base = f_base(1:floor(h{1}.N_sample/2));

figure(3)
plot(f_base,power_spectra)
xlabel('f[Hz]')
xlim([0 Fs/2])
ylabel('power spectra')
title('Power spectra for channel '+string(channel)+' run '+string(run))

clear run channel ch_idx f_base power_spectra fc

%% Laplacian filter
lap = zeros(size(electrode_map));
for k = 1:n_ch-1 %we need to ecìxlude the "Status"

    %look for the channel in the map
    [ch_row, ch_col] = find(electrode_map==ch_names(k));
    
    lap(find(ch_names==ch_names(k)),k) = 1;

    non_zero_entry = [];
    %look up
    if ch_row>1
        non_zero_entry = [non_zero_entry, find(ch_names==electrode_map(ch_row-1,ch_col))];
    end
    %look sx
    if ch_col>1
        non_zero_entry = [non_zero_entry, find(ch_names==electrode_map(ch_row,ch_col-1))];
    end
    %look dx
    if ch_col<size(electrode_map,2)
        non_zero_entry = [non_zero_entry, find(ch_names==electrode_map(ch_row,ch_col+1))];
    end    
    %look down
    if ch_row<size(electrode_map,1)
        non_zero_entry = [non_zero_entry, find(ch_names==electrode_map(ch_row+1,ch_col))];
    end
    
    if isempty(non_zero_entry)
        %pass
    else
        lap(non_zero_entry,k) = -1/length(non_zero_entry);
    end
end

clear k non_zero_entry ch_col ch_row

save('laplacian32.mat','lap')

%% PSD

load("load_file/laplacian32.mat")

%power spectral density
wlength = 0.5; % seconds. Length of the internal window
pshift = 0.25; % seconds. Shift of the internal windows
wshift = 0.0625; % seconds. Shift of the external window

mlength = 1; % seconds

selected_freq = [4 30];

h_PSD = {};
PSD_signal = {};

for k = 1:n_run

    %applicazione del filtro 
    s_laplacian = s_filt{k}(:,1:(n_ch-1)) * lap;
    
    [PSD, f] = proc_spectrogram(s_laplacian, wlength, wshift, pshift, Fs, mlength);
    
    %select meaningfull frequences 
    h_PSD{k}.f = f(find(f==selected_freq(1)):find(f==selected_freq(2)));
    PSD_signal{k} = PSD(:,find(f==selected_freq(1)):find(f==selected_freq(2)),:);
    
    %recompute the Position of the event related to the PSD windows
    h_PSD{k}.EVENT.POS = proc_pos2win(h{k}.EVENT.POS, wshift*Fs, 'backward', wlength*Fs);
    h_PSD{k}.EVENT.DUR = round(h{k}.EVENT.DUR./(wshift*Fs));
    h_PSD{k}.EVENT.TYP = h{k}.EVENT.TYP;
end
clear k s_laplacian wshift wlength pshift mlength PSD f;

%% ERD visual

task_codes = [CODE.Both_Feet CODE.Both_Hand CODE.Rest];

for run = 1:n_run

    trial_span = [CODE.Fixation_cross CODE.Continuous_feedback];
    [Activity,Ck_trial] = signal_into_trial(PSD_signal{run},h_PSD{run},trial_span,task_codes);
    %[windows x frequencies x channels x trials]

    trial_span = [CODE.Fixation_cross CODE.Fixation_cross];
    [Reference,~] = signal_into_trial(PSD_signal{run},h_PSD{run},trial_span,task_codes);
    %[windows x frequencies x channels x trials]

    % for this graph i dont need the rest trials
    Activity = Activity(:,:,:,Ck_trial~=CODE.Rest);
    Reference = Reference(:,:,:,Ck_trial~=CODE.Rest);
    Ck_trial(Ck_trial==CODE.Rest) = [];
    
    % ERD (event related desync) = log(activity/reference)
    base_line = repmat(mean(Reference,1), size(Activity,1), 1, 1, 1);
    
    ERD = log(Activity./base_line); %sample x frequency x channels x trial
    
    ERD_mean_Hand = mean(ERD(:,:,:,Ck_trial==CODE.Both_Hand),4); %mean over trial
    ERD_mean_Feet = mean(ERD(:,:,:,Ck_trial==CODE.Both_Feet),4);
    
    wshift = 0.0625; % seconds. Shift of the external window in the PSD
    time_base = [0 wshift*size(Activity,1)];
    freq_base = [h_PSD{run}.f(1) h_PSD{run}.f(end)];
    
    clims = [-2 0.5];
    ch_idx = find(ch_names=='C3' | ch_names=='CZ' | ch_names=='C4');
    
    figure(3+run)
    sgtitle('ERD run: '+string(run))
    for k = 1:length(ch_idx)
        subplot(2,length(ch_idx),k)
        imagesc(ERD_mean_Hand(:,:,ch_idx(k))','XData',time_base,'YData',freq_base)
        title(join(['ERD Both Hand' ch_names(ch_idx(k))]))
        xlabel('time[s]')
        ylabel('frequency[Hz]')
        set(gca,'YDir','normal','CLim',clims)
        colormap hot
        colorbar
        
        subplot(2,length(ch_idx),k+length(ch_idx))
        imagesc(ERD_mean_Feet(:,:,ch_idx(k))','XData',time_base,'YData',freq_base)
        title(join(['ERD Both Feet' ch_names(ch_idx(k))]))
        xlabel('time[s]')
        ylabel('frequency[Hz]')
        set(gca,'YDir','normal','CLim',clims)
        colormap hot
        colorbar
    end
end
clear ERD ERD_mean_Feet ERD_mean_Hand freq_base ch_idx clims base_line Ck_trial
clear k run Activity Reference task_codes time_base trial_span wshift

%% Feature extraction

th = 0.70;

FS_matrix = [];
PSD_reshape = {};
Ck_win = {};
figure(7)
for run = 1:n_run

    [feature_table,reshape_idx,FS_matrix(:,:,run),PSD_reshape{run},Ck_win{run}] = fisher_matrix(th,PSD_signal{run},h_PSD{run},CODE,ch_names);
    
    subplot(1, n_run, run)
    imagesc(FS_matrix(:,:,run),'XData',h_PSD{run}.f,'YData',[1:size(PSD_signal{run},3)])
    xticks(h_PSD{run}.f)
    yticks(1:size(PSD_signal{run},3))
    set(gca,'YTickLabel', ch_names)
    title('Features matrix run: '+string(run))
    ylabel('Channels')
    xlabel('frequency[Hz]')
    colormap default
    colorbar

    disp('Feature Run: '+string(run))
    disp(feature_table)
end

FS_matrix_avg = mean(FS_matrix,3);
figure(8)
imagesc(FS_matrix_avg,'XData',h_PSD{1}.f,'YData',[1:size(PSD_signal{run},3)])
xticks(h_PSD{1}.f)
yticks(1:size(PSD_signal{1},3))
set(gca,'YTickLabel', ch_names)
title('Features matrix average')
ylabel('Channels')
xlabel('frequency[Hz]')
colormap default
colorbar

clear run feature_table reshape_idx FS_matrix_avg FS_matrix

%% by visual inspection
features = ["CZ" 10; "C3" 10; "C4" 10; "C3" 8]

feature_ch = [];
feature_f = [];
for k = 1:size(features,1)
    feature_ch(k) = find(ch_names==features(k,1));
    feature_f(k) = find(h_PSD{1}.f==str2double(features(k,2)));
end

reshape_idx = sub2ind([length(h_PSD{1}.f), n_ch-1],feature_f,feature_ch);

clear k features feature_ch feature_f

%% Training set

trn_set = [];
true_label = [];
trial.POS = [0];
trial.TYP = [];
trial.DUR = [];
for run = 1:n_run
    trn_set = [trn_set; PSD_reshape{run}];
    true_label = [true_label; Ck_win{run}];

    trial.POS = [trial.POS; trial.POS(end)+h_PSD{run}.EVENT.POS]; 
    trial.TYP = [trial.TYP; h_PSD{run}.EVENT.TYP];
    trial.DUR = [trial.DUR; h_PSD{run}.EVENT.DUR];
end
trn_set = trn_set(:,reshape_idx);
trial.POS(1) = [];

% the training set is made up all the cue-->end continuous feedback period
% in order to have the information about trials we need to manipulate the
% struct trial
trial.info = 'trial start and label refers to cue-->end of continuous feedback period';
trial.start = [];
trial.label = zeros(n_trial,1);

trial_start = trial.POS(trial.TYP==CODE.Both_Feet | trial.TYP==CODE.Both_Hand | trial.TYP==CODE.Rest); %cue
trial_end = trial.POS(trial.TYP==CODE.Continuous_feedback) + trial.DUR(trial.TYP==CODE.Continuous_feedback); %end of continuous feedback
trial_len = trial_end - trial_start; %length of each trial

n_trial = length(trial_len);

for k = 1:n_trial
    trial.start = [trial.start; 1; zeros(trial_len(k)-1,1)];
    trial.label(k) = trial.TYP(find(trial.POS==trial_start(k)));
end


clear run k trial_start trial_end trial_length 

%% Gaussian classifier
n_feature = size(trn_set,2);

both_feet_data = trn_set(true_label==CODE.Both_Feet,:);
both_hand_data = trn_set(true_label==CODE.Both_Hand,:);

% training one gmm for each feature
gmm_both_feet = cell(1,n_feature);
gmm_both_hand = cell(1,n_feature);

for k = 1:n_feature
    [gmm_both_feet{k},~,~] = gmm(both_feet_data(:,k),[1 2 3 4],1000,'CrossValidation','Display');
    [gmm_both_hand{k},~,~] = gmm(both_hand_data(:,k),[1 2 3 4],1000,'CrossValidation','Display');
end
%% multidimensional mixture
option = statset('MaxIter',1000,'TolFun',1e-6);
mdgmm_both_feet = fitgmdist(both_feet_data,4,"Options",option);

option = statset('MaxIter',1000,'TolFun',1e-6);
mdgmm_both_hand = fitgmdist(both_hand_data,4,"Options",option);

disp('Multi dimensional mixture model trained')

%% feature distribution
base = 0:0.01:10;
figure(9)
sgtitle('Features distribution + gmm for each feature')
for k = 1:n_feature
    subplot(1, n_feature, k)
    histogram(both_feet_data(:,k),100,'Normalization','pdf','FaceColor','b')
    hold on
    plot(base, pdf(gmm_both_feet{k},base'),'b-','LineWidth',2)
    histogram(both_hand_data(:,k),100,'Normalization','pdf','FaceColor','r')
    plot(base, pdf(gmm_both_hand{k},base'),'r-','LineWidth',2)
    hold off
    ylim([0 18])
    xlim([0 3])
    title('Feature '+string(k))
end
legend('Both Feet','Both Feet model', 'Both Hand', 'Both Hand model')
clear base k both_hand_data both_feet_data

%% Classififer output
% binary classifier (both hand and both feet)
n_sample = size(trn_set,1);

trn_pp = [];

for k = 1:n_sample
    data = trn_set(k,:);
    both_hand_likelihood = [];
    both_feet_likelihood = [];
    for m = 1:n_feature
        both_feet_likelihood(m) = pdf(gmm_both_feet{m},data(m));
        both_hand_likelihood(m) = pdf(gmm_both_hand{m},data(m));
    end
    raw_likelihood = [prod(both_feet_likelihood), prod(both_hand_likelihood)];

    trn_pp(k,:) = raw_likelihood./sum(raw_likelihood);
end
clear raw_likelihood both_hand_likelihood both_feet_likelihood

f = 16;%Hz
time_base = [0:n_sample-1]/f;
label_plot(true_label==CODE.Both_Feet) = 0.9;
label_plot(true_label==CODE.Rest) = 0.5;
label_plot(true_label==CODE.Both_Hand) = 0.1;

figure(10)
plot(time_base, trn_pp(:,1),'wo','MarkerFaceColor','w','MarkerSize',0.5)
hold on
plot(time_base(true_label==CODE.Both_Hand), label_plot(true_label==CODE.Both_Hand), 'r.','LineWidth',2)
plot(time_base(true_label==CODE.Both_Feet), label_plot(true_label==CODE.Both_Feet), 'b.','LineWidth',2)
plot(time_base(true_label==CODE.Rest), label_plot(true_label==CODE.Rest), 'g.','LineWidth',2)
hold off
xlim([time_base(1), time_base(time_base==120)])
xlabel('sec')
ylabel('prob')
title('classifier output probabilities')
legend('raw trn output','both hand','both feet','rest')

figure(11)
sgtitle('Data distribution')
subplot(131)
histogram(trn_pp(true_label==CODE.Both_Hand,1),100,'Normalization',"pdf",'FaceColor',"#D95319")
ylim([0 12])
title('Both hand class')

subplot(132)
histogram(trn_pp(true_label==CODE.Both_Feet,1),100,'Normalization',"pdf",'FaceColor',"#0072BD")
ylim([0 12])
title('Both feet class')

subplot(133)
histogram(trn_pp(true_label==CODE.Rest,1),100,'Normalization',"pdf",'FaceColor',"#77AC30")
ylim([0 12])
title('Rest class')

save('output_file/raw_pp_output_1.mat','trn_pp','true_label','trial')

%% with the multidimensional
trn_pp_md = [];

for k = 1:n_sample
    data = trn_set(k,:);

    both_feet_likelihood = pdf(mdgmm_both_feet,data);
    both_hand_likelihood = pdf(mdgmm_both_hand,data);

    raw_likelihood = [both_feet_likelihood, both_hand_likelihood];

    trn_pp_md(k,:) = raw_likelihood./sum(raw_likelihood);
end
clear raw_likelihood both_hand_likelihood both_feet_likelihood

figure(12)
plot(time_base, trn_pp_md(:,1),'wo','MarkerFaceColor','w','MarkerSize',0.5)
hold on
plot(time_base(true_label==CODE.Both_Hand), label_plot(true_label==CODE.Both_Hand), 'r.','LineWidth',2)
plot(time_base(true_label==CODE.Both_Feet), label_plot(true_label==CODE.Both_Feet), 'b.','LineWidth',2)
plot(time_base(true_label==CODE.Rest), label_plot(true_label==CODE.Rest), 'g.','LineWidth',2)
hold off
xlim([time_base(1), time_base(time_base==120)])
xlabel('sec')
ylabel('prob')
title('classifier output probabilities MD')
legend('raw trn output','both hand','both feet','rest')


figure(13)
sgtitle('Data distribution MD')
subplot(131)
histogram(trn_pp_md(true_label==CODE.Both_Hand,1),100,'Normalization',"pdf",'FaceColor',"#D95319")
ylim([0 12])
title('Both hand class')

subplot(132)
histogram(trn_pp_md(true_label==CODE.Both_Feet,1),100,'Normalization',"pdf",'FaceColor',"#0072BD")
ylim([0 12])
title('Both feet class')

subplot(133)
histogram(trn_pp_md(true_label==CODE.Rest,1),100,'Normalization',"pdf",'FaceColor',"#77AC30")
ylim([0 12])
title('Rest class')

save('output_file/raw_pp_output_1_md.mat','trn_pp_md','true_label','trial')