% Coupled estimation and regularization of local regularity and local power
% with manually chosen regularization parameters (image)
%
% from B. Pascal, N. Pustelnik, P. Abry: How Joint Fractal Features
% Estimation and Texture Segmentation can be cast into a Strongly Convex
% Optimization Problem?, (2019) arXiv:1910.05246

function x = coupled_manual(L,lbd,alph)
    
    % inputs  - L: log-leaders and their estimated covariance structure
    %         - maxit: maximum number of iterations for BFGS algorithm
    %         - lbd, alph: regularization parameters
    %
    % outputs - x.h: regularized local regularity obtained with chosen
    %         hyperparameters
    %         - x.v: regularized local power obtained with chosen
    %         hyperparameters
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    % Chosen regularization parameters
    prox.lambda = lbd;
    prox.alpha = alph;
    
    % Minimization of the ROF functional
    [v,h] = PA_PDc(L,prox);
    x.h = h;
    x.v = v;
    x.meth = 'C';
    
end