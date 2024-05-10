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
