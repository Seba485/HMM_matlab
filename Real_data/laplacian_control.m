clear
close all
clclapalacian

load("laplacian32.mat")

load("lapmask_antneuro_32.mat")

if sum((lapmask == lap)==0)~=0
    disp('Sono Diversi')
else
    disp('Sono Uguali')
end
