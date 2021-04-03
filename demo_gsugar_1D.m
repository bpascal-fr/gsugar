% B. Pascal
% February 2020
% Fractal process segmentation from fractal features with
% Automatic tuning of hyperparameters from a Stein-based approach
%
% from
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246
% and
% - B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

close all
clear variables
clear global variables
clc


addpath(genpath('./'))

%% GENERATION OF THE PIECEWISE Fractional Gaussian Field TO BE SEGMENTED

% Signal of length 2^N
N = 10;

% Local regularities H
H0 = 0.5; H1 = 0.9;

% Local variances 
Var0 = 0.6; Var1 = 1.1;

% Different masks
[H, Var, MASK] = mask_1D(N,H0,H1,Var0,Var1); % central window

% Piecewise monofractal
X = fgn1D_piecewise(N,H,Var);
figure(1); clf
subplot(231); plot(1:2^N,MASK,'linewidth',2); title('Mask','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on
subplot(232); plot(1:2^N,X,'linewidth',2); title('Signal','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on

%% MULTI-SCALE ANALYSIS - LEADERS COEFFICIENTS

JJ = 1:6;              % range of scales (default 1:3)

% Compute leaders and perform ordinary linear regression
L_X = multiscale_analysis_1D(X,  JJ);
figure(1);
subplot(233); plot(1:2^N,L_X.h_LR,'linewidth',2); title('$\widehat{h}^{\mathrm{LR}}$','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on

%% BFGS MINIMIZATION USING SUGAR -- ROF FUNCTIONAL

% Automatic search for best lambda
% maxit = 10;
% lbd_in.l = 1e-1;
[x_rof_opt,lbd_rof_opt] = bfgs_rof_gsugar_1D(L_X);
% [x_rof_opt,lbd_rof_opt] = bfgs_rof_gsugar_1D(L_X,maxit,lbd_in);
    % maxit: max. number iterations (defaut 20)
    % lbd_in.l: initial lambda
    % x_opt.h: estimate of local regularity with lowest estimated risk
    % lbd_opt.l: optimal hyperparameter lambda
    
% Segmentation and score
K = 2;      % number of regions
[seg_rof_opt,Th_rof_opt]= trof(x_rof_opt.h,K);
    % Th_opt: thresholded optimal estimate of local regularity 
    % seg_opt: obtained segmentation for optimal hyperparameters
perf_rof = score(seg_rof_opt,MASK);
    % percentage of well-classified pixels
    
display_result_1D(x_rof_opt,Th_rof_opt)
G_rof = global_analysis(X, L_X, seg_rof_opt);





%% BFGS MINIMIZATION USING SUGAR -- JOINT FUNCTIONAL

% Automatic search for best (lambda_h,lambda_v)
% maxit = 10;
% lbd_in.h = 1e-2; lbd_in.v = 1e-2;
[x_j_opt,lbd_j_opt] = bfgs_joint_gsugar_1D(L_X);
% [x_j_opt,lbd_j_opt] = bfgs_joint_gsugar_1D(L_X,maxit,lbd_in);
    % maxit: max. number iterations (defaut 20)
    % lbd_in.h: initial lambda_h (defaut (5.5) of the article)
    % lbd_in.v: initial lambda_v (defaut (5.5) of the article)
    % x_opt.h: estimate of local regularity with lowest estimated risk
    % x_opt.v: estimate of local power with lowest estimated risk
    % lbd_opt.h: optimal hyperparameter lambda_h
    % lbd_opt.v: optimal hyperparameter lambda_v
    
% Segmentation and score
K = 2;      % number of regions
[seg_j_opt,Th_j_opt]= trof(x_j_opt.h,K);
    % Th_opt: thresholded optimal estimate of local regularity 
    % seg_opt: obtained segmentation for optimal hyperparameters
perf_j = score(seg_j_opt,MASK);
    % percentage of well-classified pixels
    
display_result_1D(x_j_opt,Th_j_opt)
G_j = global_analysis(X, L_X, seg_j_opt);


%% BFGS MINIMIZATION USING SUGAR -- COUPLED FUNCTIONAL

% Automatic search for best (lambda_h,lambda_v)
% maxit = 10;
% lbd_in.l = 1e-2; lbd_in.a = 1e-2;
[x_c_opt,lbdc_opt] = bfgs_coupled_gsugar_1D(L_X);
% [x_opt,lbd_opt] = bfgs_coupled_gsugar_1D(L_X,maxit,lbd_in);
    % maxit: max. number iterations (defaut 20)
    % lbd_in.l: initial lambda
    % lbd_in.a: initial alpha
    % x_opt.h: estimate of local regularity with lowest estimated risk
    % x_opt.v: estimate of local power with lowest estimated risk
    % lbd_opt.l: optimal hyperparameter lambda
    % lbd_opt.a: optimal hyperparameter alpha
    
% Segmentation and score
K = 2;      % number of regions
[seg_c_opt,Th_c_opt]= trof(x_c_opt.h,K);
    % Th_opt: thresholded optimal estimate of local regularity 
    % seg_opt: obtained segmentation for optimal hyperparameters
perf_c = score(seg_c_opt,MASK);
    % percentage of well-classified pixels
    
display_result_1D(x_c_opt,Th_c_opt)
G_c = global_analysis(X, L_X, seg_c_opt);
