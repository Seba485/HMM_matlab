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

real_data = real;
%% DATA
f = 16;%Hz

if real_data
    %load the static classifier output
    load("/home/sebastiano/HMM_matlab/Real_data/output_file/raw_pp_output_1_md.mat")
    trn_pp = trn_pp_md(:,1);
else
    %simulated data
    [trn_pp, true_label, trial] = create_simulated_data(CODE,10000,10,f);
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

if real_data
    %tst_pp = tst_pp(:,1);
    %just for this time the training is the same of the test
    tst_pp = trn_pp;
else
    %simulated data
    [tst_pp, true_label, trial] = create_simulated_data(CODE,5000,5,f);
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

%% FORWARD ALGORITHM between buffer FIFO VERSION + FRAMEWORK

% hmm parameters and initialization
dt_buffer = 1; %sec
buffer_len = dt_buffer*f;
fifo = [];
initial_posterior = ones(N_state,1).*1/N_state;
hmm_pred.row_out = initial_posterior.*ones(N_state,buffer_len-1); %array of posterior
hmm_pred.label = CODE.Rest*ones(buffer_len-1,1); %label point wise
hmm_pred.pp = 0.5*ones(buffer_len-1,1); %probability of classification
hmm_pred.trial_span = []; %prediction only for the current trial
hmm_pred.trial = []; %predicted trial label


% Dynamic framework parameters
y_stable_dyn = 0.5; %were the classifier has to to stai during INC (intentional non control)
w = 0.3; %boudaries of the conservative part [y_stable-w y_stable+w]  
psi = 0.5; %heigth of the potential valley (conservative part of the Free force)
chi = 0.5; %reactivity of the sistem
phi = 0.4; %weight on the previous framework value [0 1]
parameters = [y_stable_dyn w psi chi phi];
dyn.output = 0.5*ones(buffer_len-1,1);
dyn.pred = []; %point wise prediction
dyn.trial = []; %trial prediction
dyn.trial_span = []; %point wise prediction for the current trial
th_dyn = [0.2 0.8]; %threshold for the framework

% Exponential framework parameters
y_stable_exp = 0.5; %were the classifier has to to stai during INC (intentional non control)
alpha = 0.9; %weigth on previous value of the framework
exp.output = 0.5*ones(buffer_len-1,1);
exp.pred = []; %point wise prediction
exp.trial = []; %trial prediction
exp.trial_span = []; %point wise prediction for the current trial
th_exp = [0.2 0.8]; %thresh old for the framework


%hard classification --> start with rest
hmm_hard = [];
state_name = ["both hand", "both feet", "rest"];
class_code = [CODE.Both_Hand, CODE.Both_Feet, CODE.Rest];
x_code = [0 1 0.5];


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
        posterior = likelihood.*(hmm_pred.row_out(:,end)'*T)';
     
        %normalization    
        posterior = posterior./sum(posterior);

        hmm_pred.row_out = [hmm_pred.row_out, posterior];

        % fifo update
        fifo(1) = [];
        
        % hard classification
        [max_pp,max_idx] = max(posterior);

        hmm_pred.pp = [hmm_pred.pp; max_pp];
        hmm_pred.label = [hmm_pred.label; class_code(max_idx)];
        hmm_pred.trial_span = [hmm_pred.trial_span; class_code(max_idx)];

        x = x_code(max_idx);       
        hmm_hard = [hmm_hard; x];

        % hmm trial classification
        if length(hmm_pred.trial_span)>1 && (trial.start(k)==1 || k==n_sample) %this means that we concluded at least a trial
            % trial classification
            n_hmm.bh = sum(hmm_pred.trial_span==CODE.Both_Hand);
            n_hmm.bf = sum(hmm_pred.trial_span==CODE.Both_Feet);
            n_hmm.r = sum(hmm_pred.trial_span==CODE.Rest);
           
            [~,idx] = max([n_hmm.bh, n_hmm.bf, n_hmm.r]);
            hmm_pred.trial = [hmm_pred.trial; class_code(idx)];

            % reset counter
            n_hmm.bh = 0;
            n_hmm.bf = 0;
            n_hmm.r = 0;
            hmm_pred.trial_span = [];
        end
        

        % dynamic framework
        if trial.start(k)==1 || k==n_sample %new tiral
            if ~isempty(dyn.trial_span) %this means that we concluded a trial
                n_dyn.bh = sum(dyn.trial_span==CODE.Both_Hand);
                n_dyn.bf = sum(dyn.trial_span==CODE.Both_Feet);
                n_dyn.r = sum(dyn.trial_span==CODE.Rest);
                    
                % trial classification
                [~,idx] = max([n_dyn.bh, n_dyn.bf, n_dyn.r]);
                dyn.trial = [dyn.trial; class_code(idx)];

                % resent counter
                n_dyn.bh = 0;
                n_dyn.bf = 0;
                n_dyn.r = 0;

                dyn.trial_span = [];
            end
            y = y_stable_dyn;
        else
            y = Dynamic_framework(dyn.output(end),x,parameters);
        end
        dyn.output = [dyn.output; y];

        %point wise classification
        if y<=th_dyn(1)
            dyn.pred = [dyn.pred; class_code(1)];
            dyn.trial_span = [dyn.trial_span; class_code(1)];
        elseif y>=th_dyn(2)
            dyn.pred = [dyn.pred; class_code(2)];
            dyn.trial_span = [dyn.trial_span; class_code(2)];
        else
            dyn.pred = [dyn.pred; class_code(3)];
            dyn.trial_span = [dyn.trial_span; class_code(3)];
        end

        % exponential framework
        if trial.start(k)==1 || k == n_sample %new tiral
            if ~isempty(exp.trial_span) %this means that we concluded a trial
                n_exp.bh = sum(exp.trial_span==CODE.Both_Hand);
                n_exp.bf = sum(exp.trial_span==CODE.Both_Feet);
                n_exp.r = sum(exp.trial_span==CODE.Rest);
                    
                % trial classification
                [~,idx] = max([n_exp.bh, n_exp.bf, n_exp.r]);
                exp.trial = [exp.trial; class_code(idx)];

                % resent counter
                n_exp.bh = 0;
                n_exp.bf = 0;
                n_exp.r = 0;

                exp.trial_span = [];
            end
            y = y_stable_exp;
        else
            y = exp.output(end)*alpha + x*(1-alpha);
        end
        exp.output = [exp.output; y];
        
        %point wise classification
        if y<=th_exp(1)
            exp.pred = [exp.pred; class_code(1)];
            exp.trial_span = [exp.trial_span; class_code(1)];
        elseif y>=th_exp(2)
            exp.pred = [exp.pred; class_code(2)];
            exp.trial_span = [exp.trial_span; class_code(2)];
        else
            exp.exp = [exp.pred; class_code(3)];
            exp.trial_span = [exp.trial_span; class_code(3)];
        end
    end
end

% HMM accuracy (point accuracy)
hmm_pred.accuracy.overall = 100*sum(hmm_pred.label==true_label)/length(true_label);
hmm_pred.accuracy.bh = 100*sum(hmm_pred.label(true_label==CODE.Both_Hand)==CODE.Both_Hand)/sum(true_label==CODE.Both_Hand);
hmm_pred.accuracy.bf = 100*sum(hmm_pred.label(true_label==CODE.Both_Feet)==CODE.Both_Feet)/sum(true_label==CODE.Both_Feet);
hmm_pred.accuracy.r = 100*sum(hmm_pred.label(true_label==CODE.Rest)==CODE.Rest)/sum(true_label==CODE.Rest);

n_trial = sum(trial.start==1);
% HMM trial accuracy
% the trial is classified as the most frequent class in it
hmm_pred.trial_acc.overall = 100*sum(hmm_pred.trial==trial.label)/n_trial;
hmm_pred.trial_acc.bh = 100*sum(hmm_pred.trial(trial.label==CODE.Both_Hand)==CODE.Both_Hand)/sum(trial.label==CODE.Both_Hand);
hmm_pred.trial_acc.bf = 100*sum(hmm_pred.trial(trial.label==CODE.Both_Feet)==CODE.Both_Feet)/sum(trial.label==CODE.Both_Feet);
hmm_pred.trial_acc.r = 100*sum(hmm_pred.trial(trial.label==CODE.Rest)==CODE.Rest)/sum(trial.label==CODE.Rest);
    

% Dynamic framework trial accuracy
dyn.accuracy.overall = 100*sum(dyn.trial==trial.label)/n_trial;
dyn.accuracy.bh = 100*sum(dyn.trial(trial.label==CODE.Both_Hand)==CODE.Both_Hand)/sum(trial.label==CODE.Both_Hand);
dyn.accuracy.bf = 100*sum(dyn.trial(trial.label==CODE.Both_Feet)==CODE.Both_Feet)/sum(trial.label==CODE.Both_Feet);
dyn.accuracy.r = 100*sum(dyn.trial(trial.label==CODE.Rest)==CODE.Rest)/sum(trial.label==CODE.Rest);

% Exponential framework trial accuracy 
exp.accuracy.overall = 100*sum(exp.trial==trial.label)/n_trial;
exp.accuracy.bh = 100*sum(exp.trial(trial.label==CODE.Both_Hand)==CODE.Both_Hand)/sum(trial.label==CODE.Both_Hand);
exp.accuracy.bf = 100*sum(exp.trial(trial.label==CODE.Both_Feet)==CODE.Both_Feet)/sum(trial.label==CODE.Both_Feet);
exp.accuracy.r = 100*sum(exp.trial(trial.label==CODE.Rest)==CODE.Rest)/sum(trial.label==CODE.Rest);

% plot
%--------------------------------------------------------------------------
clear state_plot
state_plot(hmm_pred.label==CODE.Both_Feet) = 1;
state_plot(hmm_pred.label==CODE.Both_Hand) = 0;
state_plot(hmm_pred.label==CODE.Rest) = 0.5;

clear label_plot
label_plot(true_label==CODE.Both_Feet) = 1;
label_plot(true_label==CODE.Rest) = 0.5;
label_plot(true_label==CODE.Both_Hand) = 0;

time_base = [0:n_sample-1]/f;
y_lim = [-0.1 1.1];
figure(8)
sgtitle('HMM inference on '+string(dt_buffer)+'sec buffer FIFO')
subplot(311)
plot(time_base, tst_pp,'wo','MarkerFaceColor','w','MarkerSize',1)
xlim([time_base(1), time_base(end)])
ylim(y_lim)
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
ylim(y_lim)
xlabel('t[sec]')
hold on
plot(time_base,state_plot,'g--','LineWidth',0.5)
hold off
legend('True lable','Predicted label')
title('HMM output')

subplot(313)
plot(time_base(hmm_pred.label==CODE.Both_Hand),hmm_pred.pp(hmm_pred.label==CODE.Both_Hand),'ro','MarkerFaceColor','r','MarkerSize',3)
xlim([time_base(1), time_base(end)])
ylim(y_lim)
xlabel('t[sec]')
hold on
plot(time_base(hmm_pred.label==CODE.Both_Feet),hmm_pred.pp(hmm_pred.label==CODE.Both_Feet),'bo','MarkerFaceColor','b','MarkerSize',3)
plot(time_base(hmm_pred.label==CODE.Rest),hmm_pred.pp(hmm_pred.label==CODE.Rest),'go','MarkerFaceColor','g','MarkerSize',3)
hold off
legend('Both hand pp', 'Both feet pp', 'Rest pp')
title('Probability output hmm')

figure(9)
subplot(221)
bar([1:4], [hmm_pred.accuracy.overall, hmm_pred.accuracy.bh, hmm_pred.accuracy.bf, hmm_pred.accuracy.r])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('HMM Accuracy')
grid on

subplot(222)
bar([1:4], [hmm_pred.trial_acc.overall, hmm_pred.trial_acc.bh, hmm_pred.trial_acc.bf, hmm_pred.trial_acc.r])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('HMM trial Accuracy')
grid on

subplot(223)
bar([1:4], [dyn.accuracy.overall, dyn.accuracy.bh, dyn.accuracy.bf, dyn.accuracy.r])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('HMM + Dynamic framework trial accuracy')
grid on

subplot(224)
bar([1:4], [exp.accuracy.overall, exp.accuracy.bh, exp.accuracy.bf, exp.accuracy.r])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('HMM + Exponential framework trial accuracy')
grid on

figure(10)
plot(time_base,label_plot,'w-','LineWidth',1.5)
xlim([time_base(1), time_base(end)])
ylim(y_lim)
xlabel('t[sec]')
hold on
plot(time_base,state_plot,'g--','LineWidth',0.5)
plot(time_base,dyn.output,'r-','LineWidth',1)
plot(time_base,exp.output,'b-','LineWidth',1)
stem(time_base(trial.start==1),ones(n_trial,1),'m-','LineWidth',2,'Marker','none')
hold off
legend('True lable','hmm label','Dynamic fw','Exponential fw','trial')
title('Framework comparison')

%--------------------------------------------------------------------------


        















        

    
