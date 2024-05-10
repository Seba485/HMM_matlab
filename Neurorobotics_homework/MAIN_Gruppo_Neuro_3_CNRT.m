clear all
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
%% Data inspection

% Load OffLine data
dataset_path = '/home/sebastiano/HMM_matlab/calibration_data';
subject_folder = '20231213_d7';
type = 'calibration';
[s,t,h] = LoadAndConcatenate_data(dataset_path,subject_folder,type);

 % run = 'ai6.20180316.154006.offline.mi.mi_bhbf.gdf';
 % [s, h] = sload([dataset_path '/' subject_folder '/' run]);
 % h.N = size(s,1);
 % t = [0:1/h.SampleRate:(h.N-1)/h.SampleRate]';
 % h.ch_names = ["Fz","FC3","FC1","FCz","FC2","FC4","C3","C1","Cz","C2","C4","CP3","CP1","CPz","CP2","CP4","Ref"];

% visual inspection of the central channels
ch = [7 9 11];

figure('Name','Visual inspection')
sgtitle(['Concatenated ' type ' signal'])
for k = 1:length(ch)
    subplot(length(ch), 1, k)
    plot(t,s(:,ch(k)))
    grid on
    title(h.ch_names(ch(k)))
    xlabel('t[s]')
    ylabel('\muV')
    xlim([t(1) t(end)])
end

%% Artifact removal

% by visual inspection of the channels of interest is possible to see how
% the artifact are common for all the channels, so only une channel (Cz) is
% taken as reference for artifact detection
ch_ref = 9;
limit = 200;%uV
[s,t,h,count] = Artifact_Trial_removal(s,t,h,ch_ref,limit,CODE);
disp('Trial removed due artifact corruption = '+string(count));

if count>=1
    figure('Name','Visual inspection')
    sgtitle(['Concatenated ' type ' signal after trial removal'])
    for k = 1:length(ch)
        subplot(length(ch), 1, k)
        plot(t,s(:,ch(k)))
        grid on
        title(h.ch_names(ch(k)))
        xlabel('t[s]')
        ylabel('\muV')
        xlim([t(1) t(end)])
    end
end

%one channel with events
ch_show = 9;
Labeling_Graph(h,s,t,ch_show);


%% LOGARITHMIC BAND POWER
trial_span = [CODE.Fixation_cross CODE.Continuous_feedback];
task_codes = [CODE.Both_Feet CODE.Both_Hand];

mu_band = [9 13];
[Log_avg_mu] = Grand_average_log_power(s,h,mu_band,trial_span,task_codes);

beta_band = [18 24];
[Log_avg_beta] = Grand_average_log_power(s,h,beta_band,trial_span,task_codes);

t_trial = [0:size(Log_avg_mu,1)-1]*(t(2)-t(1));
figure('Name','Log Power Average')
sgtitle('Log Power Grand Average')
for k = 1:length(ch)
    subplot(2,length(ch),k)
    plot(t_trial,Log_avg_mu(:,ch(k),1))
    hold on
    plot(t_trial,Log_avg_mu(:,ch(k),2))
    legend('both feet','both hand')
    title(join(['\mu band [9 13]Hz ch:' h.ch_names(ch(k))]))
    xlabel('t[s]')
    ylabel('Amplitude [Log(\muV^{2})]')
    ylim([0 5])
    grid on
    
    subplot(2,length(ch),k+length(ch))
    plot(t_trial,Log_avg_beta(:,ch(k),1))
    hold on
    plot(t_trial,Log_avg_beta(:,ch(k),2))
    legend('both feet','both hand')
    title(join(['\beta band [18 24]Hz ch:' h.ch_names(ch(k))]))
    xlabel('t[s]')
    ylabel('Amplitude [Log(\muV^{2})]')
    ylim([0 5])
    grid on
end

%% PSD --> ERD
load("laplacian16.mat")

selected_freq = [4, 48];

[PSD_signal, h_PSD] = PSD_computation(s, h, selected_freq, lap);

task_codes = [CODE.Both_Feet CODE.Both_Hand];

trial_span = [CODE.Fixation_cross CODE.Continuous_feedback];
[Activity,Ck_trial] = Signal_into_trial(PSD_signal,h_PSD,trial_span,task_codes);
%[windows x frequencies x channels x trials]

trial_span = [CODE.Fixation_cross CODE.Fixation_cross];
[Reference,~] = Signal_into_trial(PSD_signal,h_PSD,trial_span,task_codes);
%[windows x frequencies x channels x trials]

% ERD (event related desync) = log(activity/reference)
base_line = repmat(mean(Reference,1), size(Activity,1), 1, 1, 1);

ERD = log(Activity./base_line); %sample x frequency x channels x trial

ERD_mean_Hand = mean(ERD(:,:,:,Ck_trial==CODE.Both_Hand),4); %mean over trial
ERD_mean_Feet = mean(ERD(:,:,:,Ck_trial==CODE.Both_Feet),4);

wshift = 0.0625; % seconds. Shift of the external window in the PSD
time_base = [0 wshift*size(Activity,1)];
freq_base = [h_PSD.f(1) h_PSD.f(end)];

clims = [-2 0.5];
ch = [7 9 11];

figure('Name','ERD')
sgtitle('ERD')
for k = 1:length(ch)
    subplot(2,length(ch),k)
    imagesc(ERD_mean_Hand(:,:,ch(k))','XData',time_base,'YData',freq_base)
    title(join(['ERD Both Hand' h.ch_names(ch(k))]))
    xlabel('time[s]')
    ylabel('frequency[Hz]')
    set(gca,'YDir','normal','CLim',clims)
    colormap hot
    colorbar
    
    subplot(2,length(ch),k+length(ch))
    imagesc(ERD_mean_Feet(:,:,ch(k))','XData',time_base,'YData',freq_base)
    title(join(['ERD Both Feet' h.ch_names(ch(k))]))
    xlabel('time[s]')
    ylabel('frequency[Hz]')
    set(gca,'YDir','normal','CLim',clims)
    colormap hot
    colorbar
end

%% Fisher features selection
th = 0.6;
[feature_ch,feature_freq,FS_matrix,PSD_reshape,Ck_win] = Fisher_matrix(PSD_signal,h_PSD,CODE,th);


figure('Name','Features selection')
imagesc(FS_matrix,'XData',h_PSD.f,'YData',[1:size(PSD_signal,3)])
xticks(h_PSD.f)
yticks(1:size(PSD_signal,3))
set(gca,'YTickLabel', h.ch_names)
title('Features matrix')
ylabel('Channels')
xlabel('frequency[Hz]')
colormap default
colorbar

selected_features = sub2ind(size(FS_matrix'),feature_freq,feature_ch);
ch_f_feature = [feature_ch, h_PSD.f(feature_freq)];

dim = length(selected_features);
disp('Number of selected features: '+string(dim))
disp('Selected features:')
for k = 1:dim
    disp('ch: '+string(h.ch_names(feature_ch(k)))+'  f: '+string(h_PSD.f(feature_freq(k)))+'Hz')
end

%% Features extraction
dataset = PSD_reshape(:,selected_features);

tot_tr.data = length(Ck_win);
tot_tr.hand = sum(Ck_win==CODE.Both_Hand);
tot_tr.feet = sum(Ck_win==CODE.Both_Feet);

figure('Name','Features')
sgtitle('Separability')
n=1;
for k = 1:dim
    for j = 1:dim
        if k==j
            subplot(dim,dim,n)
            plot(find(Ck_win==CODE.Both_Hand),dataset(Ck_win==CODE.Both_Hand,k),'b^')
            hold on
            plot(find(Ck_win==CODE.Both_Feet),dataset(Ck_win==CODE.Both_Feet,k),'r^')
            hold off
            xlabel('Feature '+string(k))
            set(gca,'YScale','log')
        else
            subplot(dim,dim,n)
            plot(dataset(Ck_win==CODE.Both_Hand,k),dataset(Ck_win==CODE.Both_Hand,j),'b^')
            hold on
            plot(dataset(Ck_win==CODE.Both_Feet,k),dataset(Ck_win==CODE.Both_Feet,j),'r^')
            hold off
            set(gca,'XScale','log','YScale','log')
            xlabel('Feature '+string(k))
            ylabel('Feature '+string(j))
        end
        n = n+1;
    end
end

if size(ch_f_feature,1)>=3
    figure('Name', 'Separability_3D')
    plot3(dataset(Ck_win==CODE.Both_Hand,1),dataset(Ck_win==CODE.Both_Hand,2),dataset(Ck_win==CODE.Both_Hand,3),'b^')
    hold on
    plot3(dataset(Ck_win==CODE.Both_Feet,1),dataset(Ck_win==CODE.Both_Feet,2),dataset(Ck_win==CODE.Both_Feet,3),'r^')
    hold off
    set(gca,'XScale','log','YScale','log','ZScale','log')
    xlabel('Feature 1')
    ylabel('Feature 2')
    zlabel('Feature 3')
    grid on
end

trn_set = [dataset,Ck_win];
save('dataset.mat',"trn_set")
%% Model Training

model = fitcdiscr(dataset(Ck_win~=CODE.Rest,:),Ck_win(Ck_win~=CODE.Rest),"DiscrimType","quadratic"); %decision tree

[train_pred, train_pp] = predict(model,dataset);

overall_accuracy_train = 100*sum(Ck_win==train_pred)/tot_tr.data;

Hand_accuracy_train = 100*sum(train_pred(Ck_win==CODE.Both_Hand)==CODE.Both_Hand)/tot_tr.hand;
Feet_accuracy_train = 100*sum(train_pred(Ck_win==CODE.Both_Feet)==CODE.Both_Feet)/tot_tr.feet;

figure('Name','Performances')
bar([1:3], [Hand_accuracy_train, overall_accuracy_train, Feet_accuracy_train])
xticklabels(["Both Hand" "Overall" "Both Feet"])
ylabel('Accuracy %')
ylim([0 100])
title('Training Accuracy')
grid on

model_name = ['model_' subject_folder '.mat'];
save(model_name,"model","selected_features","ch_f_feature")


first_class_trn = train_pp(Ck_win==CODE.Both_Hand,1); %true lable
second_class_trn = train_pp(Ck_win==CODE.Both_Feet,1); %true lable
rest_class_trn = train_pp(Ck_win==CODE.Rest,1); %true lable

Ck_trn = Ck_win;
trn_pp = train_pp;
save('raw_pp_output.mat', "trn_pp","Ck_trn")

%%
clear all
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
%% Test the Model on online data (test set) (stand alone part)

% Load online data
dataset_path = '/home/sebastiano/HMM_matlab/calibration_data';
subject_folder = '20231215_d7';
type = 'calibration';
[s_online,t_online,h_online] = LoadAndConcatenate_data(dataset_path,subject_folder,type);

%load the model
%model_name = ['model_' subject_folder '.mat'];
%load(model_name);

% run = 'ah7.20170613.170929.online.mi.mi_bhbf.ema.gdf';
% [s_online, h_online] = sload([dataset_path '\' subject_folder '\' run]);
% h_online.N = size(s_online,1);
% t_online = [0:1/h.SampleRate:(h.N-1)/h.SampleRate]';
% h_online.ch_names = ["Fz", "FC3","FC1","FCz","FC2","FC4","C3","C1","Cz","C2","C4","CP3","CP1","CPz","CP2","CP4","Ref"];

load("laplacian16.mat")
selected_freq = [4, 48];
%PSD
[PSD_online, h_PSD_online] = PSD_computation(s_online, h_online, selected_freq, lap);

% test set
[test_set, framework_testset, tot, trial_flag] = OnlineData_to_Testset(PSD_online,h_PSD_online,selected_features,CODE);

%% Raw test of the model

[test_pred, test_pp] = predict(model,test_set(:,1:end-1));
%[test_pred, test_pp] = predict(model,framework_testset(:,1:end-1));

overall_accuracy = 100*sum(test_set(:,end)==test_pred)/tot.test_data;

Hand_accuracy = 100*sum(test_pred(test_set(:,end)==CODE.Both_Hand)==CODE.Both_Hand)/tot.test_hand;
Feet_accuracy = 100*sum(test_pred(test_set(:,end)==CODE.Both_Feet)==CODE.Both_Feet)/tot.test_feet;

figure('Name','Performances')
bar([1:3], [Hand_accuracy, overall_accuracy, Feet_accuracy])
xticklabels(["Both Hand" "Overall" "Both Feet"])
ylabel('Accuracy %')
title('Test Accuracy on online data (only feedback period)')
ylim([0 100])
grid on

% Ck_tst = framework_testset(:,end);
% tst_pp = test_pp;
% save('raw_pp_output.mat', "tst_pp","Ck_tst",'-append')
% save('dataset.mat','test_set','-append')

%% Accumulation framework

[~, pp] = predict(model,framework_testset(:,1:end-1));
n_win = size(pp,1);
th = [0.2 0.8]; %related to the both feet trial

% Exponential framework
alpha = 0.9;
y_stable = 0.5; %were the classifier has to to stai during INC (intentional non control)

%initial decision (related to the first class)
D = 0.5*ones(n_win,1); 
for k = 2:n_win
    if any(ismember(trial_flag,k)) %new tiral
        D(k) = 0.5;
    else
        D(k) = Exponential_framework(D(k-1),pp(k,1),alpha);
    end
end 


[trial_classification, trial_accuracy] = Trial_accuracy(D,trial_flag,framework_testset(:,end),th,CODE);

trial_class_show = [];
trial_class_show(trial_classification==CODE.Both_Feet) = th(2);
trial_class_show(trial_classification==CODE.Both_Hand) = th(1);
trial_class_show(trial_classification==CODE.Rest) = (th(1)+th(2))/2;

figure('Name','Framework')
plot([1:n_win],pp(:,1),'k.');
hold on
stem(trial_flag,ones(1,length(trial_flag)),'r--','Marker','none','LineWidth',2);
plot([1:n_win],D,'b-')
plot([1:n_win],th(1)*ones(1,n_win),'g--',[1:n_win],th(2)*ones(1,n_win),'g--')
stem(trial_flag, trial_class_show, 'mo',"filled",'LineWidth',2,'LineStyle','none')
legend('Raw posterior prob','trial','Exp frame','th','','classification')
xlim([trial_flag(5) trial_flag(20)])
xlabel('windows')
ylabel('Posterior probability')
title('Exp frw Class 1 (Both Feet) trial accuracy = '+string(trial_accuracy))

%% Dynamic framework

y_stable = 0.5; %were the classifier has to to stai during INC (intentional non control)
w = 0.3; %boudaries of the conservative part [y_stable-w y_stable+w]  
psi = 0.2; %heigth of the potential valley (conservative part of the Free force)
chi = 10; %reactivity of the sistem
phi = 0.5; %weight on the previous framework value [0 1]

parameters = [y_stable w psi chi phi];

%initial decision (related to the first class)
y = 0.5*ones(n_win,1); 
for k = 2:n_win
    if any(ismember(trial_flag,k)) %new tiral
        y(k) = y_stable;
    else
        y(k) = Dynamic_framework(y(k-1),pp(k,1),parameters);
    end
end 

[trial_classification, trial_accuracy] = Trial_accuracy(y,trial_flag,framework_testset(:,end),th,CODE);

trial_class_show = [];
trial_class_show(trial_classification==CODE.Both_Feet) = th(2);
trial_class_show(trial_classification==CODE.Both_Hand) = th(1);
trial_class_show(trial_classification==CODE.Rest) = (th(1)+th(2))/2;

figure('Name','Framework')
plot([1:n_win],pp(:,1),'k.');
hold on
stem(trial_flag,ones(1,length(trial_flag)),'r--','Marker','none','LineWidth',2);
plot([1:n_win],y,'b-')
plot([1:n_win],th(1)*ones(1,n_win),'g--',[1:n_win],th(2)*ones(1,n_win),'g--')
stem(trial_flag, trial_class_show, 'mo',"filled",'LineWidth',2,'LineStyle','none')
legend('Raw posterior prob','trial','Dynamic frame','th','','classification')
xlim([trial_flag(5) trial_flag(20)])
xlabel('windows')
ylabel('Posterior probability')
title('Dynamic frw Class 1 (Both Feet) trial accuracy = '+string(trial_accuracy))

%% ONLINE LOOP Simulation (stand alone part)

load("laplacian16.mat")
ch = size(lap,1);

dataset_path = '/home/sebastiano/Neurorobotics/Progetto_Neuro/micontinuous';
subject_folder = 'control';
type = 'online';
[s_online,t_online,h_online] = LoadAndConcatenate_data(dataset_path,subject_folder,type);

model_name = ['model_' subject_folder '.mat'];
load(model_name);
n_features = length(selected_features);

%power spectral density parameters
wlength = 0.5; % seconds. Length of the external window
freq_resolution = 1/wlength; %frequency resolutio in the psd (window(t)-->smaplerate(f))
pshift = 0.25; % seconds. Shift of the internal windows
wshift = 0.0625; % seconds. Shift of the external window
mlength = 1; % seconds. amplitude of the mooving average

%buffer parameters
Fs = h_online.SampleRate; %512 Hz
buffer_len = mlength*Fs;%FIFO buffer of 1s --> 1 window in psd
chunk = 32;%sample
dt = chunk/Fs;%New data every dt

%classifier paramter
th = [0.2 0.8];

%framework parameters
y_stable = 0.5; %were the classifier has to to stai during INC (intentional non control)
w = 0.3; %boudaries of the conservative part [y_stable-w y_stable+w]  
psi = 0.2; %heigth of the potential valley (conservative part of the Free force)
chi = 10; %reactivity of the sistem
phi = 0.5; %weight on the previous framework value [0 1]

parameters = [y_stable w psi chi phi]; 

figure('Name','Framework online')
buffer = []; %buffer for data
framework = [0.5]; %initialization fo the framework
for k = 1:chunk:size(s_online,1)
    tic
    new_chunk = s_online(k:k+chunk-1,1:ch);
    if size(buffer,1)<buffer_len
        buffer(k:k+chunk-1,:) = new_chunk;
        %and wait for the buffer to complete
    else
        %shift of the values
        buffer(1:buffer_len-chunk,:) = buffer(chunk+1:buffer_len,:);
        %new values entry
        buffer(buffer_len-chunk+1:buffer_len,:) = new_chunk;
        
        %spatial filter
        buffer_lap = buffer * lap;
    
        %Power spectral density
        [buffer_PSD, f] = proc_spectrogram(buffer_lap, wlength, wshift, pshift, Fs, mlength);
        %buffer_PSD is a single window
        
        %select meaningfull frequences 
        %since all the settings of the PSD computation are know we can
        %extract directly the needed features from the singolar PSD window

        buffer_data=[];
        for j = 1:n_features
            buffer_data(j) = buffer_PSD(1,ch_f_feature(j,2)/2+1,ch_f_feature(j,1));
        end
        
        %posterio porbability
        [~, pp] = predict(model,buffer_data);
        
        y = Dynamic_framework(framework(end),pp(1),parameters);
        framework = [framework; y];
        
        elapsed_time = toc;

        num_data = length(framework);
        
        plot(num_data,pp(1),'k.')
        hold on
        plot([1:num_data],framework,'b-')
        plot([1:num_data],th(2)*ones(1,num_data),'g--',[1:num_data],th(1)*ones(1,num_data),'r--')
        legend('Raw posterior prob','Dynamic frame','th-Both Feet','th-Both Hand')
        xlabel('windows')
        ylabel('Posterior probability')
        title('Online Simulation, Dynamic framework dt='+string(elapsed_time)+'[s]')

        pause(0.2)

    end
end





