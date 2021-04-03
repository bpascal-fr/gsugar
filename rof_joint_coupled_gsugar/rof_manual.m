% Rudin-Osher-Fatemi denoising applied on the linear regression estimate
% of local regularity with manually chosen regularization parameter (image)
%
% from 
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246

function x = rof_manual(L,lbd)
    
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
    h = PA_PD(L,prox);
    x.h = h;
    x.meth = 'ROF';
    
end