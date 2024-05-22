clear
close all
clc

%task codes
CODE.Both_Hand = 773;
CODE.Both_Feet = 771;
CODE.Rest = 783;

f = 16;%Hz

load("raw_pp_output.mat") %in case of real data
trn_pp = trn_pp(:,1);

n_sample = length(trn_pp);
time_base = [0:n_sample-1]/f;

both_hand_pp = trn_pp(true_label==CODE.Both_Hand);
both_feet_pp = trn_pp(true_label==CODE.Both_Feet);
rest_pp = trn_pp(true_label==CODE.Rest);

base = [0:0.01:1]';

%% multi gaussiano

iter = 1000;
n = [1,2,3,4,5,6];
[both_hand_model,base,multi_gaus_bh] = GMM_state(both_hand_pp,n,iter,'Display','CrossValidation',5);
[both_feet_model,~,multi_gaus_bf] = GMM_state(both_feet_pp,n,iter,'Display','CrossValidation',5);
[rest_model,~,multi_gaus_rest] = GMM_state(rest_pp,n,iter,'display','CrossValidation',5);

%l'unico che non crea probemi, ma overfitta

%% single gaussian with initial values

iter = 1000;
start.mean = [0.01]';
start.sigma = [0.5];
[both_hand_model,base,mono_gaus_bh] = GMM_state(both_hand_pp,1,iter,'Display','Start',start);
start.mean = [0.9];
start.sigma = [0.5];
[both_feet_model,~,mono_gaus_bf] = GMM_state(both_feet_pp,1,iter,'Display','Start',start);
start.mean = [0.1 0.9]';
start.sigma = [0.5 0.5];
[rest_model,~,mono_gaus_rest] = GMM_state(rest_pp,2,iter,'display','Start',start);

% non fitta bene

%% gamma fit (matlab built-in)

model = fitdist(both_hand_pp,'Gamma');
gamma_bh = pdf(model,base);

model = fitdist(both_feet_pp,'Gamma');
gamma_bf = pdf(model,base);

model = fitdist(rest_pp,'Gamma');
gamma_rest = pdf(model,base);

[param_bh, conf] = gamfit(both_hand_pp);
both_hand_model = makedist("Gamma",'a',param_bh(1),'b',param_bh(2));
gamma_bh_2 = pdf(both_hand_model,base);

[param_bf, conf] = gamfit(both_feet_pp);
both_feet_model = makedist("Gamma",'a',param_bf(1),'b',param_bf(2));
gamma_bf_2 = pdf(both_feet_model,base);

[param_rest, conf] = gamfit(rest_pp);
rest_model = makedist("Gamma",'a',param_rest(1),'b',param_rest(2));
gamma_rest_2 = pdf(rest_model,base);

% non riesce a fittare distribuzioni con skewness negativa (a sinistra) e a
% zero vale infinito

%% gamma mixture model pkg
[w, alpha_bh, beta_bh] = GMMestimator(both_hand_pp,2,1000,10e-6,false);
gamma_mix_bh = 0;
for k = 1:length(w)
    gamma_mix_bh = gamma_mix_bh + w(k).*gampdf(base,alpha_bh(k),beta_bh(k));
end

[w, alpha_bf, beta_bf] = GMMestimator(both_feet_pp,2,1000,10e-6,false);
gamma_mix_bf = 0;
for k = 1:length(w)
    gamma_mix_bf = gamma_mix_bf + w(k).*gampdf(base,alpha_bf(k),beta_bf(k));
end

[w, alpha_rest, beta_rest] = GMMestimator(rest_pp,2,1000,10e-6,false);
gamma_mix_rest = 0;
for k = 1:length(w)
    gamma_mix_rest = gamma_mix_rest + w(k).*gampdf(base,alpha_rest(k),beta_rest(k));
end

%% gamma with flexible distriution mixture package 
% (si rome se si vuole usare la beta)

model_snob = snob(both_hand_pp,{'gamma',1},'k',1,'maxk',2,'varnames',{'both_feet_pp'});
figure(1)
mm_PlotModel1d(model_snob,both_hand_pp,1)
mm_Summary(model_snob)

model_snob = snob(both_feet_pp,{'gamma',1},'k',1,'maxk',2,'varnames',{'both_feet_pp'});
figure(2)
mm_PlotModel1d(model_snob,both_feet_pp,1)
mm_Summary(model_snob)

model_snob = snob(rest_pp,{'gamma',1},'k',1,'maxk',2,'varnames',{'both_feet_pp'});
figure(3)
mm_PlotModel1d(model_snob,rest_pp,1)
mm_Summary(model_snob)

% non si resce a tirarefuori dal modello i parametri
% vuole come minimo 2 componenti
% il fitting non è buono

%% beta fit (matlab built in)

confidence = 0.05;
[phat_bh,pci] = betafit(both_hand_pp,confidence);
pd = makedist("Beta",phat_bh(1),phat_bh(2));
beta_bh = pdf(pd, base);

confidence = 0.05;
[phat_bf,pci] = betafit(both_feet_pp,confidence);
pd = makedist("Beta",phat_bf(1),phat_bf(2));
beta_bf = pdf(pd, base);

confidence = 0.05;
[phat_rest,pci] = betafit(rest_pp,confidence);
pd = makedist("Beta",phat_rest(1),phat_rest(2));
beta_rest = pdf(pd, base);

% fitting migliore e sistema più semplice
% problema di saturazione --> a xero a e 1 vale infinito in tutte le
% distribuzioni --> impossibile da usare

%% plot
y_lim = [0 12];

figure(4)
t = tiledlayout('flow','TileSpacing','compact');
sgtitle('Data distribution')
nexttile
histogram(both_hand_pp,100,'Normalization',"pdf",'FaceColor',"#FFFFFF")
hold on
plot(base,multi_gaus_bh,'r-','LineWidth',2)
plot(base,mono_gaus_bh,'r--','LineWidth',2)
plot(base,gamma_bh,'g-','LineWidth',2)
plot(base,gamma_bh_2,'g--','LineWidth',2)
plot(base,gamma_mix_bh,'g-.','LineWidth',2)
plot(base,beta_bh,'b-','LineWidth',2)
ylim(y_lim)
title('Both hand class')

nexttile
histogram(both_feet_pp,100,'Normalization',"pdf",'FaceColor',"#FFFFFF")
hold on
plot(base,multi_gaus_bf,'r-','LineWidth',2)
plot(base,mono_gaus_bf,'r--','LineWidth',2)
plot(base,gamma_bf,'g-','LineWidth',2)
plot(base,gamma_bf_2,'g--','LineWidth',2)
plot(base,gamma_mix_bf,'g-.','LineWidth',2)
plot(base,beta_bf,'b-','LineWidth',2)
ylim(y_lim)
title('Both feet class')

nexttile
histogram(rest_pp,100,'Normalization',"pdf",'FaceColor',"#FFFFFF")
hold on
plot(base,multi_gaus_rest,'r-','LineWidth',2)
plot(base,mono_gaus_rest,'r--','LineWidth',2)
plot(base,gamma_rest,'g-','LineWidth',2)
plot(base,gamma_rest_2,'g--','LineWidth',2)
plot(base,gamma_mix_rest,'g-.','LineWidth',2)
plot(base,beta_rest,'b-','LineWidth',2)
ylim(y_lim)
title('Rest class')
lgd = legend('data', 'gmm', 'gaussian', 'gamma', 'gamma_2', 'gamma mix', 'beta');
lgd.Layout.Tile = 4;




%%

mu = 0;
std = 0.05;
gaussian_pd = makedist("Normal",mu,std);
base = [0:0.01:1];

pdf_plot = pdf(gaussian_pd,base');

B = 40;
B_1 = 8;
both_hand_pdf = 10*exp(-B.*(base)) + 5*exp(-B_1.*base);
both_feet_pdf = 10*exp(B.*(base-1)) + 5*exp(B_1.*(base-1));
rest_pdf = (10*exp(-B.*(base)) + 5*exp(-5.*base) + 10*exp(B.*(base-1)) + 5*exp(5.*(base-1)))./2;

figure(1)
histogram(both_hand_pp,100,'Normalization',"pdf",'FaceColor',"#FFFFFF")
hold on
plot(base,both_feet_pdf,'r-','LineWidth',2);
hold off
ylim([0 20])
xlim([0 1])


y_lim = [0 20];
figure(3)
sgtitle('Data distribution - GMM state')
subplot(131)
histogram(both_hand_pp,100,'Normalization',"pdf",'FaceColor',"#D95319")
hold on
plot(base,both_hand_pdf,'r-','LineWidth',2)
ylim(y_lim)
title('Both hand class')

subplot(132)
histogram(both_feet_pp,100,'Normalization',"pdf",'FaceColor',"#0072BD")
hold on
plot(base,both_feet_pdf,'b-','LineWidth',2)
ylim(y_lim)
title('Both feet class')

subplot(133)
histogram(rest_pp,100,'Normalization',"pdf",'FaceColor',"#77AC30")
hold on
plot(base,rest_pdf,'g-','LineWidth',2)
ylim(y_lim)
title('Rest class')

%% da copiare ed incollare in HMM_pipeline in caso si voglia testare
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

hmm_pred = 100*sum(real_Ck_pred==true_label)/length(true_label);
hmm_acc_rest = 100*sum(real_Ck_pred(true_label==CODE.Rest)==CODE.Rest)/sum(true_label==CODE.Rest);
hmm_acc_bh = 100*sum(real_Ck_pred(true_label==CODE.Both_Hand)==CODE.Both_Hand)/sum(true_label==CODE.Both_Hand);
hmm_acc_bf = 100*sum(real_Ck_pred(true_label==CODE.Both_Feet)==CODE.Both_Feet)/sum(true_label==CODE.Both_Feet);

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
title('Raw overall accuracy: ' + string(hmm_pred) + '%')

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
bar([1:4], [hmm_pred, hmm_acc_bh, hmm_acc_bf, hmm_acc_rest])
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

hmm_pred = 100*sum(Ck_pred==true_label)/length(true_label);
hmm_acc_rest = 100*sum(Ck_pred(true_label==CODE.Rest)==CODE.Rest)/sum(true_label==CODE.Rest);
hmm_acc_bh = 100*sum(Ck_pred(true_label==CODE.Both_Hand)==CODE.Both_Hand)/sum(true_label==CODE.Both_Hand);
hmm_acc_bf = 100*sum(Ck_pred(true_label==CODE.Both_Feet)==CODE.Both_Feet)/sum(true_label==CODE.Both_Feet);

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
title('Raw overall accuracy: ' + string(hmm_pred) + '%')

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
bar([1:4], [hmm_pred, hmm_acc_bh, hmm_acc_bf, hmm_acc_rest])
xticklabels(["Overall" "Both Hand" "Both Feet" "Rest"])
ylabel('Accuracy %')
ylim([0 100])
title('Test Accuracy')
grid on

%--------------------------------------------------------------------------    