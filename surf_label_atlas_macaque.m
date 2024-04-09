% SVReg: Surface-Constrained Volumetric Registration
% Copyright (C) 2019 The Regents of the University of California and the University of Southern California
% Created by Anand A. Joshi, Chitresh Bhushan, David W. Shattuck, Richard M. Leahy 
% 
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; version 2.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301,
% USA.

function surf_label_atlas_macaque(atlasbasename)

vlname=[atlasbasename,'.label.nii.gz'];
vbfc=[atlasbasename,'.bfc.nii.gz'];
sname=[atlasbasename,'.left.mid.cortex'];
sname2=[atlasbasename,'.right.mid.cortex'];

vbfc=load_nii_BIG_Lab(vbfc);
vl_new=load_nii_BIG_Lab(vlname);
ind=find((vbfc.img>=00));%&(vl_new.img<600));
[XX,YY,ZZ]=ind2sub(size(vl_new.img),ind);XX=XX-1;YY=YY-1;ZZ=ZZ-1;
dim=vl_new.hdr.dime.pixdim(2:4);
XX=XX.*dim(1);YY=YY.*dim(2);ZZ=ZZ.*dim(3);

s=readdfs([sname,'.dfs']);
s2=readdfs([sname2,'.dfs']);

F=TriScatteredInterp(XX,YY,ZZ,double(vl_new.img(ind)),'nearest');
s.labels=F(s.vertices);
writedfs([sname,'.dfs'],s);

s2.labels=F(s2.vertices);
writedfs([sname2,'.dfs'],s2);

recolor_by_label([sname,'.dfs'],atlasbasename);
refine_ROIs_atlas2([sname,'.dfs']);
recolor_by_label([sname,'.refined.dfs'],atlasbasename);  

recolor_by_label([sname2,'.dfs'],atlasbasename);
refine_ROIs_atlas2([sname2,'.dfs']);
recolor_by_label([sname2,'.refined.dfs'],atlasbasename);  
copyfile([atlasbasename,'.left.mid.cortex.refined.dfs'],[atlasbasename,'.left.mid.cortex.dfs']);
copyfile([atlasbasename,'.right.mid.cortex.refined.dfs'],[atlasbasename,'.right.mid.cortex.dfs']);

copy_attrib_colors(atlasbasename,'left','dfs');
copy_attrib_colors(atlasbasename,'right','dfs');




