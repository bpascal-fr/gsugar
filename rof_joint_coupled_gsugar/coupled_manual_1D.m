% Coupled estimation and regularization of local regularity and local power
% with manually chosen regularization parameters (image)
%
% from 
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246

function x = coupled_manual_1D(L,lbd,alph)
    
    % inputs  - L: log-leaders 
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
    
    % Minimization of the coupled functional
    [v,h] = PA_PDc_1D(L,prox);
    x.h = h;
    x.v = v;
    x.meth = 'C';
    
end