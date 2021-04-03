% B. Pascal
% February 2020
% Image segmentation from Rudin-Osher-Fatemi model
% Manual choice of regularization parameter


close all
clear variables
clear global variables
clc


addpath(genpath('./'))

%% NOISY IMAGE

% Load image
img = double(imread('cameraman.tif'));

% Noise properties
sig   = 25;   % standard deviation of the noise
l     = 1;    % correlation length

% Noisy image
noise = conv2(randn(size(img)),ones(l,l)/l,'same');
img_n  = img + sig*noise;
I.img_n = img_n;

figure(1); clf; colormap(gray)
subplot(131); imagesc(img); axis off image; title('Original','interpreter','latex','fontsize',20)
subplot(132); imagesc(img_n); axis off image; title('Noisy','interpreter','latex','fontsize',20)


%% TV DENOISING WITH MANUAL CHOICE OF REGULARIZATION PARAMETER

% Chosen regularization parameter
prox.lambda = 100;

% Minimization of the ROF functional
x_tv = PA_PDtv(I,prox);
x_tv_man.img = x_tv;

% Segmentation
K = 2;      % number of regions
[seg_tv_man,Th_tv_man]= trof(x_tv_man.img,K);
    % Th_tv_man : thresholded optimal estimate 
    % seg_tv_man: obtained segmentation for chosen hyperparameters
    
display_result_tv(x_tv_man,seg_tv_man)