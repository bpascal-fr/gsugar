% Rudin-Osher-Fatemi denoising applied on the linear regression estimate
% of local regularity with manually chosen regularization parameter (signal)
%
% from B. Pascal, N. Pustelnik, P. Abry: How Joint Fractal Features
% Estimation and Texture Segmentation can be cast into a Strongly Convex
% Optimization Problem?, (2019) arXiv:1910.05246

function x = rof_manual_1D(L,lbd)
    
    % inputs  - L: log-leaders
    %         - lbd: regularization parameter
    %
    % outputs - x.h: regularized local regularity obtained with chosen
    %         hyperparameter
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    % Chosen regularization parameter
    prox.lambda = lbd;
    
    % Minimization of the ROF functional
    h = PA_PD_1D(L,prox);
    x.h = h;
    x.meth = 'ROF';
    
end