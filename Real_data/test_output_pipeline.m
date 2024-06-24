close all
clear all
clc

dataset_path = '/home/sebastiano';
subject_folder = 'Desktop';

root = [dataset_path '/' subject_folder '/'];

%file_name = 'xx.20240606.103929.evaluation.mi_bhbf.gdf';
file_name = 'xx.20240606.105350.calibration.mi_bhbf.gdf';
[s, h] = sload([root file_name]); %signal and header

%%
%eegc3_smr_train
%wc_save_classifier
