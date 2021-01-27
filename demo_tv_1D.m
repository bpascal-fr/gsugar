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

%% NOISY PIECEWISE CONSTANT SIGNAL

% Signal properties
N        = 1e3 ;      % length of the signal
sig      = 0.05;    % standard deviation of the noise
l        = 10;       % correlation length

% Build signal
[signal_n,signal] = signal_piecewise_cst(N,sig,l);
X.signal_n = signal_n;

figure(2); clf
subplot(131); plot(1:N,signal,'linewidth',2); title('Ground truth','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on
subplot(132); plot(1:N,signal_n,'linewidth',2); title('Noisy signal','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on



%% TV DENOISING WITH MANUAL CHOICE OF REGULARIZATION PARAMETER

% Chosen regularization parameter
prox.lambda = 10;

% Minimization of the ROF functional
x_tv_man = PA_PDtv_1D(X,prox);
    
% Segmentation
K = 2;      % number of regions
[seg_tv_man,Th_tv_man]= trof(x_tv_man,K);
    % Th_tv_man : thresholded optimal estimate 
    % seg_tv_man: obtained segmentation for chosen hyperparameters
    
subplot(133); plot(1:N,x_tv_man,'linewidth',2); title('Estimates','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on; hold on
plot(1:N,Th_tv_man,'r','linewidth',2); leg = legend('Regularized',['Thresholded ($',num2str(K),'$ levels)']); leg.Interpreter = 'Latex';