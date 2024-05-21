clear
close all
clc

%task codes
CODE.Trial_start = 1;
CODE.Fixation_cross = 786;
CODE.Both_Hand = 773;
CODE.Both_Feet = 771;
CODE.Rest = 783;
CODE.Continuous_feedback = 781;
CODE.Target_hit = 897;
CODE.Target_miss = 898;

data_type = 'real';
%data_type = 'simulated';
%% DATA
f = 16;%Hz

if strcmpi(data_type,'real')
    %load the static classifier output
    load("/home/sebastiano/HMM_matlab/Real_data/output_file/raw_pp_output_1_md.mat")
    trn_pp = trn_pp_md(:,1);
else
    %simulated data
    [trn_pp, true_label] = create_simulated_data(CODE,10000,10,f);
end

n_sample = length(trn_pp);
time_base = [0:n_sample-1]/f;

both_hand_pp = trn_pp(true_label==CODE.Both_Hand);
both_feet_pp = trn_pp(true_label==CODE.Both_Feet);
rest_pp = trn_pp(true_label==CODE.Rest);

%--------------------------------------------------------------------------

label_plot(true_label==CODE.Both_Feet) = 0.9;
label_plot(true_label==CODE.Rest) = 0.5;
label_plot(true_label==CODE.Both_Hand) = 0.1;

figure(1)
plot(time_base, trn_pp,'wo','MarkerFaceColor','w','MarkerSize',1)
hold on
plot(time_base(true_label==CODE.Both_Hand), label_plot(true_label==CODE.Both_Hand), 'r.','LineWidth',2)
plot(time_base(true_label==CODE.Both_Feet), label_plot(true_label==CODE.Both_Feet), 'b.','LineWidth',2)
plot(time_base(true_label==CODE.Rest), label_plot(true_label==CODE.Rest), 'g.','LineWidth',2)
hold off
xlim([time_base(1), time_base(end)])
xlabel('sec')
ylabel('prob')
title('classifier output probabilities')
legend('raw trn output','both hand','both feet','rest')

figure(2)
y_lim = [0 12];
sgtitle('Data distribution')
subplot(131)
histogram(both_hand_pp,100,'Normalization',"pdf",'FaceColor',"#D95319")
ylim(y_lim)
title('Both hand class')

subplot(132)
histogram(both_feet_pp,100,'Normalization',"pdf",'FaceColor',"#0072BD")
ylim(y_lim)
title('Both feet class')

subplot(133)
histogram(rest_pp,100,'Normalization',"pdf",'FaceColor',"#77AC30")
ylim(y_lim)
title('Rest class')
%--------------------------------------------------------------------------

%% HMM state training
%in the training phase we are going to solve the gaussian mixture model
%that characterize every state

% HMM with 3 states, one for each class.
% each state is a exponential model that describe the distribution
% beahaviour of the static classifier output
base = [0:0.01:1]';
plot_both_hand = hmm_state(base,'both hand');
plot_both_feet = hmm_state(base,'both feet');
plot_rest = hmm_state(base,'rest');

state_name = ["both hand", "both feet", "rest"];
%save('hmm_state_exp.mat',"state_gmm","state_name");


%--------------------------------------------------------------------------
figure(3)
sgtitle('Data distribution - GMM state')
subplot(131)
histogram(both_hand_pp,100,'Normalization',"pdf",'FaceColor',"#D95319")
hold on
plot(base,plot_both_hand,'r-','LineWidth',2)
ylim(y_lim)
title('Both hand class')

subplot(132)
histogram(both_feet_pp,100,'Normalization',"pdf",'FaceColor',"#0072BD")
hold on
plot(base,plot_both_feet,'b-','LineWidth',2)
ylim(y_lim)
title('Both feet class')

subplot(133)
histogram(rest_pp,100,'Normalization',"pdf",'FaceColor',"#77AC30")
hold on
plot(base,plot_rest,'g-','LineWidth',2)
ylim(y_lim)
title('Rest class')
%--------------------------------------------------------------------------

%% HMM inference

if strcmpi(data_type,'real')
    %tst_pp = tst_pp(:,1);
    %just for this time the training is the same of the test
    tst_pp = trn_pp(:,1);
else
    %simulated data
    [tst_pp, true_label] = create_simulated_data(CODE,5000,5,f);
end


n_sample = length(tst_pp);

state_name = ["both hand", "both feet", "rest"];

N_state = length(state_name);

% Transition matrix
%for the moment is static, then it will be dynamic based on the
%TRAVERSABILITY output
T = ones(N_state,N_state)*1/N_state;

% Static probability
%the class has the same chance of occurrence
static_prob = ones(N_state,1)*1/N_state;

% Output matrix
%since the aoutput is a continous there isn't a fixed output matrix, for
%each sample the probability of emission is computed as the relative 
%likelihood (likelihood of the state/sum of all the likelihood)

%% FORWARD ALGORITHM between buffer FIFO VERSION

dt_buffer = 1; %sec
buffer_len = dt_buffer*f;

fifo = [];

initial_state_idx = 3; %Rest

initial_posterior = ones(N_state,1).*1/N_state;

hmm_output_fifo = initial_posterior.*ones(N_state,buffer_len-1);

flag = false;
for k = 1:n_sample
    %new raw output
    new_data = tst_pp(k);
    
    %fifo loading
    fifo = [fifo; new_data];

    %once the fifo is fully loaded --> hmm inference
    if length(fifo) == buffer_len
   
        %likelihood on the entire buffer
        likelihood = zeros(N_state,1);

        for m = 1:N_state
            likelihood(m) = prod(hmm_state(fifo,state_name(m)));
        end
        likelihood = likelihood./sum(likelihood); %normalization

        %posterior
        posterior = likelihood.*(hmm_output_fifo(:,end)'*T)';
     
        %normalization    
        posterior = posterior./sum(posterior);

        hmm_output_fifo = [hmm_output_fifo, posterior];

        % fifo update
        fifo(1) = [];

    end
end

[suggested_state_pp,suggested_state] = max(hmm_output_fifo);

suggested_state(1:buffer_len-1) = initial_state_idx;
Ck_pred = zeros(length(suggested_state),1);
Ck_pred(suggested_state==1) = CODE.Both_Hand;
Ck_pred(suggested_state==2) = CODE.Both_Feet;
Ck_pred(suggested_state==3) = CODE.Rest;


raw_accuracy = 100*sum(Ck_pred==true_label)/length(true_label);
raw_accuracy_rest = 100*sum(Ck_pred(true_label==CODE.Rest)==CODE.Rest)/sum(true_label==CODE.Rest);
raw_accuracy_both_hand = 100*sum(Ck_pred(true_label==CODE.Both_Hand)==CODE.Both_Hand)/sum(true_label==CODE.Both_Hand);
raw_accuracy_both_feet = 100*sum(Ck_pred(true_label==CODE.Both_Feet)==CODE.Both_Feet)/sum(true_label==CODE.Both_Feet);

%trial accuracy (senza il framework)
% the trial is classified as the most frequent class in it

n_trial = length(trial.label);
trial_start_idx = find(trial.start==1);
class_code = [CODE.Both_Hand,CODE.Both_Feet,CODE.Rest];
trial_pred = zeros(n_trial,1); %predicted trial class
for k = 1:n_trial
    if k<n_trial
        trial_span = trial_start_idx(k):trial_start_idx(k+1); %indexes
    else
        trial_span = trial_start_idx(k):n_sample; %indexes
    end
    n_bh = sum(Ck_pred(trial_span)==CODE.Both_Hand);
    n_bf = sum(Ck_pred(trial_span)==CODE.Both_Feet);
    n_r = sum(Ck_pred(trial_span)==CODE.Rest);
    [~,idx] = max([n_bh, n_bf, n_r]);
    trial_pred(k) = class_code(idx);
end
trial_accuracy = 100*sum(trial_pred==trial.label)/n_trial;
trial_acc_bh = 100*sum(trial_pred(trial.label==CODE.Both_Hand)==CODE.Both_Hand)/sum(trial.label==CODE.Both_Hand);
trial_acc_bf = 100*sum(trial_pred(trial.label==CODE.Both_Feet)==CODE.Both_Feet)/sum(trial.label==CODE.Both_Feet);
trial_acc_r = 100*sum(trial_pred(trial.label==CODE.Rest)==CODE.Rest)/sum(trial.label==CODE.Rest);
    

% Dynamic framework
y_stable = 0.5; %were the classifier has to to stai during INC (intentional non control)
w = 0.3; %boudaries of the conservative part [y_stable-w y_stable+w]  
psi = 0.4; %heigth of the potential valley (conservative part of the Free force)
chi = 0.5; %reactivity of the sistem
phi = 0.4; %weight on the previous framework value [0 1]

parameters = [y_stable w psi chi phi];

hmm_hard = zeros(n_sample,1);
hmm_hard(Ck_pred==CODE.Both_Feet) = 1;
hmm_hard(Ck_pred==CODE.Both_Hand) = 0;
hmm_hard(Ck_pred==CODE.Rest) = 0.5;

dynamic_fw = zeros(n_sample,1); 
for k = 1:n_sample
    if trial.start(k)==1 %new tiral
        dynamic_fw(k) = y_stable;
    else
        dynamic_fw(k) = Dynamic_framework(dynamic_fw(k-1),hmm_hard(k),parameters);
    end
end 

th = [0.2 0.8];
n_trial = length(trial.label);
trial_pred_dyn = zeros(n_trial,1); %predicted trial class
for k = 1:n_trial
    if k<n_trial
        trial_span = trial_start_idx(k):trial_start_idx(k+1); %indexes
    else
        trial_span = trial_start_idx(k):trial_start_idx(end); %indexes
    end
    n_bh = sum(dynamic_fw(trial_span)<=th(1));
    n_bf = sum(dynamic_fw(trial_span)>=th(2));
    n_r = sum(dynamic_fw(trial_span)>th(1) & dynamic_fw(trial_span)<th(2));
    [~,idx] = max([n_bh, n_bf, n_r]);
    trial_pred_dyn(k) = class_code(idx);
end
trial_accuracy_dyn = 100*sum(trial_pred_dyn==trial.label)/n_trial;
trial_acc_bh_dyn = 100*sum(trial_pred_dyn(trial.label==CODE.Both_Hand)==CODE.Both_Hand)/sum(trial.label==CODE.Both_Hand);
trial_acc_bf_dyn = 100*sum(trial_pred_dyn(trial.label==CODE.Both_Feet)==CODE.Both_Feet)/sum(trial.label==CODE.Both_Feet);
trial_acc_r_dyn = 100*sum(trial_pred_dyn(trial.label==CODE.Rest)==CODE.Rest)/sum(trial.label==CODE.Rest);

% Exponential framework
y_stable = 0.5; %were the classifier has to to stai during INC (intentional non control)
alpha = 0.9; %erigth on previous value of the framework

parameters = [y_stable w psi chi phi];

hmm_hard = zeros(n_sample,1);
hmm_hard(Ck_pred==CODE.Both_Feet) = 1;
hmm_hard(Ck_pred==CODE.Both_Hand) = 0;
hmm_hard(Ck_pred==CODE.Rest) = 0.5;

exp_fw = zeros(n_sample,1); 
for k = 1:n_sample
    if trial.start(k)==1 %new tiral
        exp_fw(k) = y_stable;
    else
        exp_fw(k) = exp_fw(k-1)*alpha + hmm_hard(k)*(1-alpha);
    end
end 

th = [0.2 0.8];
n_trial = length(trial.label);
trial_pred_exp = zeros(n_trial,1); %predicted trial class
for k = 1:n_trial
    if k<n_trial
        trial_span = trial_start_idx(k):trial_start_idx(k+1); %indexes
    else
        trial_span = trial_start_idx(k):n_sample; %indexes
    end
    n_bh = sum(exp_fw(trial_span)<=th(1));
    n_bf = sum(exp_fw(trial_span)>=th(2));
    n_r = sum(exp_fw(trial_span)>th(1) & dynamic_fw(trial_span)<th(2));
    [~,idx] = max([n_bh, n_bf, n_r]);
    trial_pred_exp(k) = class_code(idx);
end
trial_accuracy_exp = 100*sum(trial_pred_exp==trial.label)/n_trial;
trial_acc_bh_exp = 100*sum(trial_pred_exp(trial.label==CODE.Both_Hand)==CODE.Both_Hand)/sum(trial.label==CODE.Both_Hand);
trial_acc_bf_exp = 100*sum(trial_pred_exp(trial.label==CODE.Both_Feet)==CODE.Both_Feet)/sum(trial.label==CODE.Both_Feet);
trial_acc_r_exp = 100*sum(trial_pred_exp(trial.label==CODE.Rest)==CODE.Rest)/sum(trial.label==CODE.Rest);



%--------------------------------------------------------------------------
clear state_plot
state_plot(Ck_pred==CODE.Both_Feet) = 0.9;
state_plot(Ck_pred==CODE.Both_Hand) = 0.1;
state_plot(Ck_pred==CODE.Rest) = 0.5;

clear label_plot
label_plot(true_label==CODE.Both_Feet) = 0.9;
label_plot(true_label==CODE.Rest) = 0.5;
label_plot(true_label==CODE.Both_Hand) = 0.1;

time_base = [0:n_sample-1]/f;
figure(8)
sgtitle('HMM inference on '+string(dt_buffer)+'sec buffer FIFO')
subplot(311)
plot(time_base, tst_pp,'wo','MarkerFaceColor','w','MarkerSize',1)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base(true_label==CODE.Both_Hand), label_plot(true_label==CODE.Both_Hand), 'r.','LineWidth',2)
plot(time_base(true_label==CODE.Both_Feet), label_plot(true_label==CODE.Both_Feet), 'b.','LineWidth',2)
plot(time_base(true_label==CODE.Rest), label_plot(true_label==CODE.Rest), 'g.','LineWidth',2)
hold off
legend('Raw test output','Both hand','Both feet','Rest')
title('Test data - True label')

subplot(312)
plot(time_base,label_plot,'w-','LineWidth',1.5)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base,state_plot,'g--','LineWidth',0.5)
plot(time_base,dynamic_fw,'r-','LineWidth',1)
hold off
legend('True lable','Predicted label','Dynamic framework')
title('Raw overall accuracy: ' + string(raw_accuracy) + '%')

subplot(313)
plot(time_base(Ck_pred==CODE.Both_Hand),suggested_state_pp(Ck_pred==CODE.Both_Hand),'ro','MarkerFaceColor','r','MarkerSize',3)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base(Ck_pred==CODE.Both_Feet),suggested_state_pp(Ck_pred==CODE.Both_Feet),'bo','MarkerFaceColor','b','MarkerSize',3)
plot(time_base(Ck_pred==CODE.Rest),suggested_state_pp(Ck_pred==CODE.Rest),'go','MarkerFaceColor','g','MarkerSize',3)
hold off
legend('Both hand pp', 'Both feet pp', 'Rest pp')
title('Probability output hmm')

figure(9)
bar([1:4], [raw_accuracy, raw_accuracy_both_hand, raw_accuracy_both_feet, raw_accuracy_rest])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('Test Accuracy FIFO')
grid on

figure(10)
bar([1:4], [trial_accuracy, trial_acc_bh, trial_acc_bf, trial_acc_r])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('Test TRIAL Accuracy FIFO')
grid on

figure(11)
bar([1:4], [trial_accuracy_dyn, trial_acc_bh_dyn, trial_acc_bf_dyn, trial_acc_r_dyn])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('Test TRIAL Accuracy FIFO + Dynamic Framework')
grid on

figure(12)
bar([1:4], [trial_accuracy_exp, trial_acc_bh_exp, trial_acc_bf_exp, trial_acc_r_exp])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('Test TRIAL Accuracy FIFO + Exponential Framework')
grid on


figure(13)
plot(time_base,label_plot,'w-','LineWidth',1.5)
xlim([time_base(1), time_base(time_base==60)])
xlabel('t[sec]')
hold on
plot(time_base,state_plot,'g--','LineWidth',0.5)
plot(time_base,dynamic_fw,'r-','LineWidth',1)
plot(time_base,exp_fw,'b-','LineWidth',1)
stem(time_base(trial.start==1),ones(n_trial,1),'m-','LineWidth',2,'Marker','none')
hold off
legend('True lable','Predicted label','Dynamic framework','trial')
title('Raw overall accuracy: ' + string(raw_accuracy) + '%')

%--------------------------------------------------------------------------

%% FORWARD ALGORITHM between buffer

dt_buffer = 1; %sec
overlap = 0.5; %[0 1]
buffer_len = dt_buffer*f;

buffer_old = [];
buffer_new = [];

initial_state_idx = 3; %Rest

initial_posterior = ones(N_state,1).*1/N_state;

hmm_output_buffer = initial_posterior;
hmm_output_idx = [1];
for k = 1:n_sample
    %new raw output
    new_data = tst_pp(k);

    buffer_new_len = length(buffer_new);

    %first buffer loading
    if buffer_new_len<buffer_len
        buffer_new = [buffer_new; new_data];
    end
    
    %every time the followiong buffer is full --> hmm inference
    if buffer_new_len == buffer_len

        %likelihood on the entire buffer
        likelihood_new = zeros(N_state,1);

        for m = 1:N_state
            %likelihood_new(m) = prod(pdf(state_gmm{m},buffer_new));
            likelihood_new(m) = prod(hmm_state(buffer_new,state_name(m)));
        end
        likelihood_new = likelihood_new./sum(likelihood_new); %normalization

        if isempty(buffer_old) %we are in the first step
            
            posterior_new = likelihood_new.*(initial_posterior'*T)';
        
        else %for the followiong steps
            
            posterior_old = hmm_output_buffer(:,end).*likelihood_old;

            posterior_new = likelihood_new.*(posterior_old'*T)';
        end
            
        posterior_new = posterior_new./sum(posterior_new); %normalization

        hmm_output_buffer = [hmm_output_buffer, posterior_new];

        % buffer update
        buffer_old = buffer_new;
        likelihood_old = likelihood_new;
        buffer_new(1:floor(buffer_len*overlap)) = [];

        % output baseline
        hmm_output_idx = [hmm_output_idx; k];
    end
end

[suggested_state_pp,suggested_state] = max(hmm_output_buffer);

suggested_state(1) = initial_state_idx;
Ck_pred = zeros(length(suggested_state),1);
Ck_pred(suggested_state==1) = CODE.Both_Hand;
Ck_pred(suggested_state==2) = CODE.Both_Feet;
Ck_pred(suggested_state==3) = CODE.Rest;

% this hmm give an uptput every dt_buffer*overlap*f sample so 
% the label predicted stll the same until a new classification occurs 
real_Ck_pred = [];
pred_label = CODE.Rest;
for k = 1:length(true_label)
    if ismember(k,hmm_output_idx)
        pred_idx = find(hmm_output_idx==k);
        pred_label = Ck_pred(pred_idx);
    end
    real_Ck_pred = [real_Ck_pred; pred_label];
end

raw_accuracy = 100*sum(real_Ck_pred==true_label)/length(true_label);
raw_accuracy_rest = 100*sum(real_Ck_pred(true_label==CODE.Rest)==CODE.Rest)/sum(true_label==CODE.Rest);
raw_accuracy_both_hand = 100*sum(real_Ck_pred(true_label==CODE.Both_Hand)==CODE.Both_Hand)/sum(true_label==CODE.Both_Hand);
raw_accuracy_both_feet = 100*sum(real_Ck_pred(true_label==CODE.Both_Feet)==CODE.Both_Feet)/sum(true_label==CODE.Both_Feet);

%--------------------------------------------------------------------------
clear state_plot
state_plot(real_Ck_pred==CODE.Both_Feet) = 0.9;
state_plot(real_Ck_pred==CODE.Both_Hand) = 0.1;
state_plot(real_Ck_pred==CODE.Rest) = 0.5;

clear label_plot
label_plot(true_label==CODE.Both_Feet) = 0.9;
label_plot(true_label==CODE.Rest) = 0.5;
label_plot(true_label==CODE.Both_Hand) = 0.1;

time_base = [0:n_sample-1]/f;
hmm_base = hmm_output_idx./f;
% classifier output
figure(6)
sgtitle('HMM inference on '+string(dt_buffer)+'sec buffer '+string(overlap*100)+'% overlap')
subplot(311)
plot(time_base, tst_pp,'wo','MarkerFaceColor','w','MarkerSize',1)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base(true_label==CODE.Both_Hand), label_plot(true_label==CODE.Both_Hand), 'r.','LineWidth',2)
plot(time_base(true_label==CODE.Both_Feet), label_plot(true_label==CODE.Both_Feet), 'b.','LineWidth',2)
plot(time_base(true_label==CODE.Rest), label_plot(true_label==CODE.Rest), 'g.','LineWidth',2)
hold off
legend('Raw test output','Both hand','Both feet','Rest')
title('Test data - True label')

subplot(312)
plot(time_base,label_plot,'w-','LineWidth',1.5)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base,state_plot,'g--','LineWidth',0.5)
hold off
legend('True lable','Predicted label')
title('Raw overall accuracy: ' + string(raw_accuracy) + '%')

subplot(313)
plot(hmm_base(Ck_pred==CODE.Both_Hand),suggested_state_pp(Ck_pred==CODE.Both_Hand),'ro','MarkerFaceColor','r','MarkerSize',3)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(hmm_base(Ck_pred==CODE.Both_Feet),suggested_state_pp(Ck_pred==CODE.Both_Feet),'bo','MarkerFaceColor','b','MarkerSize',3)
plot(hmm_base(Ck_pred==CODE.Rest),suggested_state_pp(Ck_pred==CODE.Rest),'go','MarkerFaceColor','g','MarkerSize',3)
hold off
legend('Both hand pp', 'Both feet pp', 'Rest pp')
title('Probability output hmm')

figure(7)
bar([1:4], [raw_accuracy, raw_accuracy_both_hand, raw_accuracy_both_feet, raw_accuracy_rest])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('Test Accuracy')
grid on

%--------------------------------------------------------------------------

%% FORWARD algorithm
% probability of being in state Si at time t given the previous observation
% and the model. P(o1,o2,...,ot, St=Si | model)
%in this case a buffer of 2 sample is created in order to infer what is the
%most probable state that have produced the current data sample based on
%the previous state and the transfer matrix 

hmm_output = zeros(N_state,1);

initial_state_idx = 3; %Rest
buffer_len = 2; %column vector
%only after the first buffer_len iterations ill get the prediction with that history length
buffer = [];

hmm_output = zeros(N_state,2);
hmm_output(:,1) = static_prob'; %initialization

for k = 1:n_sample-1
    new_data = tst_pp(k);
    if length(buffer)<buffer_len
        buffer = [buffer; new_data];
    else
        buffer = buffer(2:end);
        buffer = [buffer; new_data];
    end
    
    hmm_output(:,k+1) = HMM_forward_step(buffer,T,hmm_output(:,k+1-length(buffer)));
   
end

[suggested_state_pp,suggested_state] = max(hmm_output);

suggested_state(1) = initial_state_idx;
Ck_pred = zeros(length(suggested_state),1);
Ck_pred(suggested_state==1) = CODE.Both_Hand;
Ck_pred(suggested_state==2) = CODE.Both_Feet;
Ck_pred(suggested_state==3) = CODE.Rest;

raw_accuracy = 100*sum(Ck_pred==true_label)/length(true_label);
raw_accuracy_rest = 100*sum(Ck_pred(true_label==CODE.Rest)==CODE.Rest)/sum(true_label==CODE.Rest);
raw_accuracy_both_hand = 100*sum(Ck_pred(true_label==CODE.Both_Hand)==CODE.Both_Hand)/sum(true_label==CODE.Both_Hand);
raw_accuracy_both_feet = 100*sum(Ck_pred(true_label==CODE.Both_Feet)==CODE.Both_Feet)/sum(true_label==CODE.Both_Feet);

%--------------------------------------------------------------------------
clear state_plot
state_plot(Ck_pred==CODE.Both_Feet) = 0.9;
state_plot(Ck_pred==CODE.Both_Hand) = 0.1;
state_plot(Ck_pred==CODE.Rest) = 0.5;

clear label_plot
label_plot(true_label==CODE.Both_Feet) = 0.9;
label_plot(true_label==CODE.Rest) = 0.5;
label_plot(true_label==CODE.Both_Hand) = 0.1;

time_base = [0:n_sample-1]/f;
% classifier output
figure(4)
sgtitle('HMM inference on single sample')
subplot(311)
plot(time_base, tst_pp,'wo','MarkerFaceColor','w','MarkerSize',1)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base(true_label==CODE.Both_Hand), label_plot(true_label==CODE.Both_Hand), 'r.','LineWidth',2)
plot(time_base(true_label==CODE.Both_Feet), label_plot(true_label==CODE.Both_Feet), 'b.','LineWidth',2)
plot(time_base(true_label==CODE.Rest), label_plot(true_label==CODE.Rest), 'g.','LineWidth',2)
hold off
legend('Raw test output','Both hand','Both feet','Rest')
title('test data - true label')

subplot(312)
plot(time_base,label_plot,'w-','LineWidth',1.5)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base,state_plot,'g--','LineWidth',0.5)
hold off
legend('True lable','Predicted label')
title('Raw overall accuracy: ' + string(raw_accuracy) + '%')

subplot(313)
plot(time_base(Ck_pred==CODE.Both_Hand),suggested_state_pp(Ck_pred==CODE.Both_Hand),'ro','MarkerFaceColor','r','MarkerSize',1)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base(Ck_pred==CODE.Both_Feet),suggested_state_pp(Ck_pred==CODE.Both_Feet),'bo','MarkerFaceColor','b','MarkerSize',1)
plot(time_base(Ck_pred==CODE.Rest),suggested_state_pp(Ck_pred==CODE.Rest),'go','MarkerFaceColor','g','MarkerSize',1)
hold off
legend('Both hand pp', 'Both feet pp', 'Rest pp')
title('Probability output hmm')

figure(5)
bar([1:4], [raw_accuracy, raw_accuracy_both_hand, raw_accuracy_both_feet, raw_accuracy_rest])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('Test Accuracy')
grid on

%--------------------------------------------------------------------------    
        















        

    
