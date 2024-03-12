clc;clear all;close all;restoredefaultpath;

addpath(genpath('/home/ajoshi/Projects/svreg/src'));
addpath(genpath('/home/ajoshi/Projects/svreg/3rdParty'));
addpath(genpath('/home/ajoshi/Projects/svreg/MEX_Files'));
addpath(genpath('/home/ajoshi/Projects/svreg/dev'));


subbasename = '/home/ajoshi/Projects/macbse/NMT_v2.1_sym_05mm_brainsuite/NMT_v2.1_sym_05mm';

charm_atlas = '/home/ajoshi/Projects/macbse/NMT_v2.1_sym_05mm_brainsuite/CHARM_in_NMT_v2.1_sym_05mm.nii.gz';

vlname=[charm_atlas];
vl_new=load_nii_BIG_Lab(vlname);
vl_new.img=vl_new.img(:,:,:,1,2);
save_untouch_nii_gz(vl_new,[subbasename,'.label.nii.gz'])


nii2uint8('/deneb_disk/macaque_atlas_data/macaque_hemi_atlas/NMT_v2.1_sym_05mm.bfc.nii.gz','/deneb_disk/macaque_atlas_data/macaque_hemi_atlas/NMT.nii.gz',1)

left_inner = readdfs([subbasename,'.left.inner.cortex.dfs']);
left_pial = readdfs([subbasename,'.left.pial.cortex.dfs']);
left_mid = left_inner;
left_mid.vertices = (left_pial.vertices + left_inner.vertices)/2;
writedfs([subbasename,'.left.mid.cortex.dfs'], left_mid);

right_inner = readdfs([subbasename,'.right.inner.cortex.dfs']);
right_pial = readdfs([subbasename,'.right.pial.cortex.dfs']);
right_mid = right_inner;
right_mid.vertices = (right_pial.vertices + right_inner.vertices)/2;
writedfs([subbasename,'.right.mid.cortex.dfs'], right_mid);


smooth_surf_hierarchy([subbasename,'.right.mid.cortex.dfs'],10);
smooth_surf_hierarchy([subbasename,'.left.mid.cortex.dfs'],10);


for jj=1:10
    s=readdfs(sprintf('%s.left.mid.cortex_smooth%d.dfs',subbasename,jj));
    s1.vertices=s.vertices;s1.attributes=s.attributes;s1.faces=[];
    writedfs(sprintf('%s.left.mid.cortex_smooth%d.dfs',subbasename,jj),s1);
    
    s=readdfs(sprintf('%s.right.mid.cortex_smooth%d.dfs',subbasename,jj));
    s1.vertices=s.vertices;s1.attributes=s.attributes;s1.faces=[];
    writedfs(sprintf('%s.right.mid.cortex_smooth%d.dfs',subbasename,jj),s1);
end


surf_label_atlas_macaque(subbasename);



make_roilist_file(subbasename);
