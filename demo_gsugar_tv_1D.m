% B. Pascal
% February 2020
% Signal segmentation from Rudin-Osher-Fatemi model
% Automatic tuning of hyperparameters from a Stein-based approach
%
% from
% - B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434


close all
clear variables
clear global variables
clc


addpath(genpath('./'))

%% NOISY PIECEWISE CONSTANT SIGNAL

% Signal properties
N        = 1e3 ;      % length of the signal
sig      = 0.05;    % standard deviation of the noise
l        = 10;       % correlation length

% Build signal
[signal_n,signal] = signal_piecewise_cst(N,sig,l);

figure(2); clf
subplot(131); plot(1:N,signal,'linewidth',2); title('Ground truth','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on
subplot(132); plot(1:N,signal_n,'linewidth',2); title('Noisy signal','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on


%% ESTIMATED COVARIANCE MATRIX

% Expected correlation length
l_est = 10;

% Estimation of correlations
S       = xcorr(signal_n)/N - mean(signal_n)^2;
S       = S(N-l_est+1:N+l_est-1);
D       = var(signal_n);

% Prepare the variable for bfgs_sugar
X.signal_n = signal_n;
X.S     = S;
X.D     = D;

%% BFGS MINIMIZATION USING SUGAR -- TV DENOISING FUNCTIONAL

% Automatic search for best lambda
% maxit = 10;
% lbd_in.l = 1e-1;
[x_tv_opt,lbd_tv_opt] = bfgs_tv_gsugar_1D(X);
% [x_tv_opt,lbd_tv_opt] = bfgs_tv_gsugar_1D(X,maxit,lbd_in);
    % maxit:          max. number iterations (defaut 20)
    % lbd_in.l:       initial lambda
    % x_opt.signal:   estimate of the signal with lowest estimated risk
    % lbd_opt.l:      optimal hyperparameter lambda
    
% Segmentation
K = 2;      % number of regions
[seg_tv_opt,Th_tv_opt]= trof(x_tv_opt.signal,K);
    % Th_tv_opt: thresholded optimal estimate 
    % seg_tv_opt: obtained segmentation for optimal hyperparameters
    
subplot(133); plot(1:N,x_tv_opt.signal,'linewidth',2); title('Estimates','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on; hold on
plot(1:N,Th_tv_opt,'r','linewidth',2); leg = legend('Regularized',['Thresholded ($',num2str(K),'$ levels)']); leg.Interpreter = 'Latex';