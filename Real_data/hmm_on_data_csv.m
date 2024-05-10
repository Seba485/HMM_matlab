clear 
close all
clc

T = readtable('load_file/data.csv');

sep_char = ',';
pp = zeros(length(T.smrbci),1);
dynamic_frw = zeros(length(T.dynamical),1);
exponential_frw = zeros(length(T.dynamical),1);
for elem = 1:length(T.smrbci)
    %raw probability
    temp_string = T.smrbci{elem}(2:find(T.smrbci{elem}==sep_char)-1);
    pp(elem) = str2double(temp_string);
    %dyaniamical framework
    temp_string = T.dynamical{elem}(2:find(T.dynamical{elem}==sep_char)-1);
    dynamic_frw(elem) = str2double(temp_string);
    %exponential framework
    temp_string = T.exponential{elem}(2:find(T.exponential{elem}==sep_char)-1);
    exponential_frw(elem) = str2double(temp_string);
end
    
f = 16;%Hzreal_data
time_base = [0:length(pp)-1]/f;

figure(12)
plot(time_base, pp,'wo','MarkerFaceColor','w','MarkerSize',1);
xlim([time_base(1), time_base(end)])
xlabel('sec')
ylabel('prob')
title('classifier output probabilities')


% load('/home/sebastiano/HMM_matlab/HMM_code/hmm_state_beta.mat') 
% %state_gmm --> cell array with the gaussian state
% %state_name --> corresponding state

tst_pp = pp;

%% HMM definition
CODE.Both_Hand = 773;
CODE.Both_Feet = 771;
CODE.Rest = 783;

state_name = ["both hand", "both feet", "rest"];

n_sample = length(tst_pp);

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

        %log likelihood on the entire buffer
        likelihood_new = zeros(N_state,1);
        for m = 1:N_state
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

real_Ck_pred = [];
pred_label = CODE.Rest;
for k = 1:length(tst_pp)
    if ismember(k,hmm_output_idx)
        pred_idx = find(hmm_output_idx==k);
        pred_label = Ck_pred(pred_idx);
    end
    real_Ck_pred = [real_Ck_pred; pred_label];
end

% NO LABEL AVAILABLE
% raw_accuracy = 100*sum(Ck_pred==true_label(hmm_output_idx))/length(hmm_output_idx);
% raw_accuracy_rest = 100*sum(Ck_pred(true_label(hmm_output_idx)==CODE.Rest)==CODE.Rest)/sum(true_label(hmm_output_idx)==CODE.Rest);
% raw_accuracy_both_hand = 100*sum(Ck_pred(true_label(hmm_output_idx)==CODE.Both_Hand)==CODE.Both_Hand)/sum(true_label(hmm_output_idx)==CODE.Both_Hand);
% raw_accuracy_both_feet = 100*sum(Ck_pred(true_label(hmm_output_idx)==CODE.Both_Feet)==CODE.Both_Feet)/sum(true_label(hmm_output_idx)==CODE.Both_Feet);

%--------------------------------------------------------------------------
state_plot(real_Ck_pred==CODE.Both_Feet) = 0.9;
state_plot(real_Ck_pred==CODE.Both_Hand) = 0.1;
state_plot(real_Ck_pred==CODE.Rest) = 0.5;

% NO LABEL AVAILABLE
% label_plot(true_label==CODE.Both_Feet) = 0.9;
% label_plot(true_label==CODE.Rest) = 0.5;
% label_plot(true_label==CODE.Both_Hand) = 0.1;

hmm_base = hmm_output_idx./f;
% classifier output
figure(6)
sgtitle('HMM inference on '+string(dt_buffer)+'sec buffer '+string(overlap*100)+'% overlap')
subplot(311)
plot(time_base, tst_pp,'wo','MarkerFaceColor','w','MarkerSize',1)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base,state_plot,'g--','LineWidth',0.5)
% plot(time_base(true_label==CODE.Both_Hand), label_plot(true_label==CODE.Both_Hand), 'r.','LineWidth',2)
% plot(time_base(true_label==CODE.Both_Feet), label_plot(true_label==CODE.Both_Feet), 'b.','LineWidth',2)
% plot(time_base(true_label==CODE.Rest), label_plot(true_label==CODE.Rest), 'g.','LineWidth',2)
% hold off
legend('Raw test output')%,'Both hand','Both feet','Rest')
title('Test data - True label (NOT AVAILABLE)')

subplot(312)
plot(time_base,state_plot,'g--','LineWidth',0.5)
xlim([time_base(1), time_base(end)])
xlabel('t[sec]')
hold on
plot(time_base,exponential_frw,'b','LineWidth',0.5)
plot(time_base,dynamic_frw,'r','LineWidth',0.5)
% plot(time_base,label_plot,'w-','LineWidth',1.5)
% hold off
legend('Predicted label','Exponential frw','Dynamic frw')%,'True lable')
title('Raw overall accuracy: NOT AVAILABLE')% + string(raw_accuracy) + '%')

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

% NOT AVAILABLE
% figure(7)
% bar([1:4], [raw_accuracy, raw_accuracy_both_hand, raw_accuracy_both_feet, raw_accuracy_rest])
% xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
% ylabel('Accuracy %')
% ylim([0 100])
% title('Test Accuracy')
% grid on

%--------------------------------------------------------------------------