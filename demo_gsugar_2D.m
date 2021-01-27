% B. Pascal
% February 2020
% Texture segmentation from fractal features
% Automatic tuning of hyperparameters from a Stein-based approach


close all
clear variables
clear global variables
clc


addpath(genpath('./'))

%% GENERATION OF THE PIECEWISE Fractional Gaussian Field TO BE SEGMENTED

% Image of size 2^N X 2^N
N = 8;

% Local regularities H
H0 = 0.5; H1 = 0.9;

% Local variances 
Var0 = 0.6; Var1 = 1.1;

% Different masks
[H, Var, MASK] = mask_ellipse(N,H0,H1,Var0,Var1);         % one central ellipse
% [H, Var, MASK] = mask_ellipse_PIECES(N,H0,H1,Var0,Var1);  % four ellipses
% [H, Var, MASK] = mask_rectangle(N,H0,H1,Var0,Var1);       % central rectangle

% Piecewise monofractal
X = fgn2D_piecewise(N,H,Var);
figure(1); clf; colormap(gray)
subplot(231); imagesc(MASK); axis off image; title('Mask','interpreter','latex','fontsize',20)
subplot(232); imagesc(X); axis off image; title('Texture','interpreter','latex','fontsize',20)

%% MULTI-SCALE ANALYSIS - LEADERS COEFFICIENTS

JJ = 1:3;              % range of scales (default 1:3)

% Compute leaders and perform ordinary linear regression
L_X = multiscale_analysis(X,  JJ);
figure(1)
subplot(233); imagesc(L_X.h_LR); axis off image; title('$\widehat{h}^{\mathrm{LR}}$','interpreter','latex','fontsize',20)


%% BFGS MINIMIZATION USING SUGAR -- ROF FUNCTIONAL

% Automatic search for best lambda for rof algorithm
% maxit = 10;
% lbd_in.l = 1e-2;
[x_rof_opt,lbd_rof_opt] = bfgs_rof_gsugar(L_X);
% [x_rof_opt,lbd_rof_opt] = bfgs_rof_gsugar(L_X, maxit,lbd_in);
    % maxit: max. number iterations (defaut 20)
    % lbd_in.h: initial lambda (defaut (5.5) of the article)
    % x_opt.h: estimate of local regularity with lowest estimated risk
    % lbd_opt.l: optimal hyperparameter lambda
    
% Segmentation and score
K = 2;      % number of regions
[seg_rof_opt,Th_rof_opt]= trof(x_rof_opt.h,K);
    % Th_opt: thresholded optimal estimate of local regularity 
    % seg_opt: obtained segmentation for optimal hyperparameters
perf_rof = score(seg_rof_opt,MASK);
    % percentage of well-classified pixels
    
display_result(x_rof_opt,seg_rof_opt)
G_rof = global_analysis(X, L_X, seg_rof_opt);

%% BFGS MINIMIZATION USING SUGAR -- JOINT FUNCTIONAL

% Automatic search for best (lambda_h,lambda_v) for joint algorithm
% maxit = 10;
% lbd_in.h = 1e-2; lbd_in.v = 1e-2;
[x_j_opt,lbd_j_opt] = bfgs_joint_gsugar(L_X);
% [x_j_opt,lbd_j_opt] = bfgs_joint_gsugar(L_X, maxit,lbd_in);
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
    
display_result(x_j_opt,seg_j_opt)
G_j = global_analysis(X, L_X, seg_j_opt);

%% BFGS MINIMIZATION USING SUGAR -- COUPLED FUNCTIONAL

% Automatic search for best (lambda,alpha) for coupled algorithm
% maxit = 10;
% lbd_in.l = 1e-2; lbd_in.a = 1e-2;
[x_c_opt,lbd_c_opt] = bfgs_coupled_gsugar(L_X);
% [x_c_opt,lbd_c_opt] = bfgs_coupled_gsugar(L_X, maxit,lbd_in);
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
    
display_result(x_c_opt,seg_c_opt)
G_c = global_analysis(X, L_X, seg_c_opt);