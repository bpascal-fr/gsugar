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
[L_X, C_X] = multiscale_analysis(X,  JJ);
figure(1)
subplot(233); imagesc(L_X.h_LR); axis off image; title('$\widehat{h}^{\mathrm{LR}}$','interpreter','latex','fontsize',20)


%% ROF FUNCTIONAL WITH MANUAL CHOICE OF REGULARIZATION PARAMETER

% Chosen regularization parameter
lambda_rof = 2.5;

% Minimization of the ROF functional
x_rof_man = rof_manual(L_X,lambda_rof);

% Segmentation and score
K = 2;      % number of regions
[seg_rof_man,Th_rof_man]= trof(x_rof_man.h,K);
    % Th_man: thresholded optimal estimate of local regularity 
    % seg_man: obtained segmentation for optimal hyperparameters
perf_rof = score(seg_rof_man,MASK);
    % percentage of well-classified pixels
    
display_result(x_rof_man,seg_rof_man)
G_rof = global_analysis(X, L_X, seg_rof_man);


%% JOINT FUNCTIONAL WITH MANUAL CHOICE OF REGULARIZATION PARAMETER

% Chosen regularization parameters
lambda_h_j = 10;
lambda_v_j = 2;

% Minimization of the Joint functional
x_j_man = joint_manual(L_X,lambda_h_j,lambda_v_j);

% Segmentation and score
K = 2;      % number of regions
[seg_j_man,Th_j_man]= trof(x_j_man.h,K);
    % Th_man: thresholded optimal estimate of local regularity 
    % seg_man: obtained segmentation for optimal hyperparameters
perf_j = score(seg_j_man,MASK);
    % percentage of well-classified pixels
    
display_result(x_j_man,seg_j_man)
G_j = global_analysis(X, L_X, seg_j_man);

%% COUPLED FUNCTIONAL WITH MANUAL CHOICE OF REGULARIZATION PARAMETER

% Chosen regularization parameter
lambda_c = 3.5;
alpha_c = 4.5;

% Minimization of the Joint functional
x_c_man = coupled_manual(L_X,lambda_c,alpha_c);

% Segmentation and score
K = 2;      % number of regions
[seg_c_man,Th_c_man]= trof(x_c_man.h,K);
    % Th_man: thresholded optimal estimate of local regularity 
    % seg_man: obtained segmentation for optimal hyperparameters
perf_c = score(seg_c_man,MASK);
    % percentage of well-classified pixels
    
display_result(x_c_man,seg_c_man)
G_c = global_analysis(X, L_X, seg_c_man);




