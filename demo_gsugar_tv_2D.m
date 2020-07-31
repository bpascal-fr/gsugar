% B. Pascal
% February 2020
% Texture segmentation from fractal features
% Automatic tuning of hyperparameters from a Stein-based approach


close all
clear variables
clear global variables
clc


addpath(genpath('./'))
addpath(genpath('/Users/pabry/ownCloud/PhD_Barbara/Matlab/gsugar-master/'))

%% NOISY IMAGE

% Load image
img = double(imread('cameraman.tif'));

% Noise properties
sig   = 25;   % standard deviation of the noise
l     = 1;    % correlation length

% Noisy image
noise = conv2(randn(size(img)),ones(l,l)/l,'same');
img_n  = img + sig*noise;


figure(1); clf; colormap(gray)
subplot(131); imagesc(img); axis off image; title('Original','interpreter','latex','fontsize',20)
subplot(132); imagesc(img_n); axis off image; title('Noisy','interpreter','latex','fontsize',20)

%% ESTIMATED COVARIANCE MATRIX

% Expected correlation length
l_est = 1;

% Estimation of correlations
[N1,N2] = size(img_n);
S       = xcorr2(img_n)/numel(img_n) - mean(img_n(:))^2;
S       = S(N1-l_est+1:N1+l_est-1, N2-l_est+1:N2+l_est-1);
D       = var(img_n(:));

% Prepare the variable for bfgs_sugar
I.img_n = img_n;
I.S     = S;
I.D     = D;

%% BFGS MINIMIZATION USING SUGAR -- TV DENOISING FUNCTIONAL

% Automatic search for best lambda for rof algorithm
% maxit = 10;
% lbd_in.l = 1e-2;
[x_tv_opt,lbd_tv_opt] = bfgs_tv_gsugar(I);
% [x_tv_opt,lbd_tv_opt] = bfgs_tv_gsugar(camb, maxit,lbd_in);
    % maxit:       max. number iterations (defaut 20)
    % lbd_in.h:    initial lambda (defaut (5.5) of the article)
    % x_opt.img:   estimate of the image with lowest estimated risk
    % lbd_opt.l:   optimal hyperparameter lambda
    
% Segmentation
K = 2;      % number of regions
[seg_tv_opt,Th_tv_opt]= trof(x_tv_opt.img,K);
    % Th_tv_opt: thresholded optimal estimate 
    % seg_tv_opt: obtained segmentation for optimal hyperparameters
    
display_result_tv(x_tv_opt,seg_tv_opt)