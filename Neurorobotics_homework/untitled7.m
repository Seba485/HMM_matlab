clear
close all
clc

load("dataset.mat")
%           features    label
%pattern

both_hand = trn_set(trn_set(:,end)==773,1:end-1);
both_feet = trn_set(trn_set(:,end)==771,1:end-1);

figure(1)
plot3(both_hand(:,1),both_hand(:,2),both_hand(:,4),'bo')
hold on
plot3(both_feet(:,1),both_feet(:,2),both_feet(:,4),'r+')
legend('both hand', 'both feet')
grid on
xlabel('C3 12Hz')
ylabel('C1 12Hz')
%zlabel('Cz 22Hz')
zlabel('Cz 24Hz')

option = statset('MaxIter',500,'TolFun',1e-6);
mvGMM_both_hand = fitgmdist(both_hand,4,'Options',option);
%mu --> each row are the mean of a single multivariate distribution
%sigma --> covariance matrix

option = statset('MaxIter',500,'TolFun',1e-6);
mvGMM_both_feet = fitgmdist(both_feet,4,'Options',option);

pdf_both_hand = pdf(mvGMM_both_hand,trn_set(:,1:end-1));
pdf_both_feet = pdf(mvGMM_both_feet,trn_set(:,1:end-1));

prob_both_hand = pdf_both_hand./(pdf_both_hand + pdf_both_feet);
prob_both_feet = pdf_both_feet./(pdf_both_hand + pdf_both_feet);