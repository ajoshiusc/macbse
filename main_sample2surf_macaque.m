%||AUM||
%||Shree Ganeshaya Namaha||

%Macaque load surface
clear all;close all;clc;

addpath(genpath('/home/ajoshi/Projects/svreg/dev/matlab_gifti/gifti-main'));
addpath(genpath('/home/ajoshi/Projects/svreg/src'));
addpath(genpath('/home/ajoshi/Projects/svreg/3rdParty'));
addpath(genpath('/home/ajoshi/Projects/svreg/MEX_Files'));



subid = 'sub-1001';


fmri = ['/home/ajoshi/MACAQUE_DEMO_REST/data_20_ap_vox/',subid,'/ses-01/sub-1001.results/errts.sub-1001.tproject.nii'];
lsurf = '/home/ajoshi/MACAQUE_DEMO_REST/NMT_v2.1_sym/NMT_v2.1_sym_05mm/left.mid.dfs';
rsurf = '/home/ajoshi/MACAQUE_DEMO_REST/NMT_v2.1_sym/NMT_v2.1_sym_05mm/right.mid.dfs';

v=load_nii_BIG_Lab(fmri);
sl = readdfs(lsurf);
sr = readdfs(rsurf);

datal=zeros(length(sl.vertices),size(v.img,4));
datar=zeros(length(sl.vertices),size(v.img,4));

res=v.hdr.dime.pixdim(2:4);
vimg=double(v.img);

for j=1:size(v.img,4)
   datal(:,j)=interp3(vimg(:,:,:,j), sl.vertices(:,2)/res(2) + 1,sl.vertices(:,1)/res(1) + 1, sl.vertices(:,3)/res(3) + 1);
   datar(:,j)=interp3(vimg(:,:,:,j), sr.vertices(:,2)/res(2) + 1,sr.vertices(:,1)/res(1) + 1, sr.vertices(:,3)/res(3) + 1);

end

save([subid,'.mat'],"datar","datal");


figure;
patch('vertices',sl.vertices,'faces',sl.faces,'facevertexcdata',datal(:,10),'facecolor','interp','edgecolor','none');
axis equal;axis off;
camlight;
material dull;