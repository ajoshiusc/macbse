clc;clear all;close all;restoredefaultpath;

addpath(genpath('/home/ajoshi/Projects/BrainSuite/svreg/src'));
addpath(genpath('/home/ajoshi/Projects/BrainSuite/svreg/3rdParty'));
addpath(genpath('/home/ajoshi/Projects/BrainSuite/svreg/MEX_Files'));

atlasbasename= '/home/ajoshi/projects/BrainSuite/svreg/USCBrain/USCBrain';
%subbasename = '/home/ajoshi/Desktop/SVRegError/sub-season101_ses-2_acq-MPrageHiRes/anat/sub-season102_ses-1_acq-MPrageHiRes_T1w';

%atlasbasename= '/home/ajoshi/projects/BrainSuite/svreg/BrainSuiteAtlas1/mri';
subbasename = '/home/ajoshi/Downloads/BrainSuiteTutorialCSE_Sept16/BrainSuiteTutorialCSE/svreg_23a_matlab_uscbrain/2523412';

%svreg(subbasename,atlasbasename,'-k');
% % % Compute Thickness
thicknessPVC(subbasename);
% % 
% % % Surface registration
svreg_label_surf_hemi(subbasename,atlasbasename,'right');
svreg_label_surf_hemi(subbasename,atlasbasename, 'left');
% % 
% % % Surface label refinement
  refine_ROIs2(subbasename, 'right');
  refine_ROIs2(subbasename,'left');
% % 
% % % Map thickness to atlas
  svreg_thickness2atlas(subbasename) 
% % 
% % % Volume registration
   volmap_ball(subbasename);
   volmap_ball([subbasename,'.target']);
   svreg_volreg(subbasename, [subbasename,'.target']);    
   svreg_refinements(subbasename, [subbasename,'.target']);
% % 
% % % Refine sulcal curved
 refine_sulci_hemi(subbasename,'left');
 refine_sulci_hemi(subbasename,'right');
% % 
% % 
% % % Clean intermediate curves
%  %clean_intermediate_files(subbasename);
% % 
% % 
% % % Generate statistical analysis
   generate_stats_xls(subbasename);
