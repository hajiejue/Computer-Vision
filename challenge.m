%% Computer Vision Challenge 2019

% Group number:
group_number = 56;

% Group members:
% members = {'Max Mustermann', 'Johannes Daten'};
members = {'Wudamu','Tianlin Kong','Liang Tian','Shiwei han','Bowen Ma'};

% Email-Address (from Moodle!):
% mail = {'ga99abc@tum.de', 'daten.hannes@tum.de'};
mail = {'ge46gud@mytum.de','ge82dof@mytum.de','ge46niv@mytum.de','ge46nib@mytum.de','bowen.ma@tum.de'};

%% Start timer here
tic

%% Disparity Map
% Specify path to scene folder containing img0 img1 and calib

% Here you have to choose the path twice, first select the path to scene folder
% Second time select the ground truth file.
 scene_path = uigetdir;
% Calculate disparity map and Euclidean motion
 [D, R, T] = disparity_map(scene_path);

%% Validation
 %gt_path = '';
 %Select the ground truth file
 [file,gt_path] = uigetfile('*.pfm','Select the ground truth file');
% Load the ground truth
% G = pfmread(gt_path);

fid = fopen(file); 
fscanf(fid,'%c',[1,3]);
cols = fscanf(fid,'%f',1);
rows = fscanf(fid,'%f',1);
fscanf(fid,'%f',1);
fscanf(fid,'%c',1);
G = fread(fid,[cols,rows],'single');
G(G == Inf) = 0;
G = rot90(G);
fclose(fid);
% Estimate the quality of the calculated disparity map
 p = verify_dmap(D, G);

%% Stop timer here
elapsed_time = toc;


%% Print Results
 R, T, p, elapsed_time


%% Display Disparity
imagesc(D);
colormap(jet);

