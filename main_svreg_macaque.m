%||AUM||
%||Shree Ganeshaya Namaha||

%Macaque load surface
clear all;close all;clc;

addpath(genpath('/home/ajoshi/Projects/svreg/dev/matlab_gifti/gifti-main'));
addpath(genpath('/home/ajoshi/Projects/svreg/src'));
addpath(genpath('/home/ajoshi/Projects/svreg/3rdParty'));
addpath(genpath('/home/ajoshi/Projects/svreg/MEX_Files'));

subbasename = '/home/ajoshi/Projects/macbse/data/sub-032196_ses-001_run-1_T1w';
atlasbasename = '/home/ajoshi/Projects/svreg/NMT_v2.1_sym_05mm_brainsuite/NMT_v2.1_sym_05mm';

svreg(subbasename,atlasbasename);



