% Joint estimation and regularization of local regularity and local power
% with manually chosen regularization parameters (image)
%
% from B. Pascal, N. Pustelnik, P. Abry: How Joint Fractal Features
% Estimation and Texture Segmentation can be cast into a Strongly Convex
% Optimization Problem?, (2019) arXiv:1910.05246

function x = joint_manual(L,lbd_h,lbd_v)
    
    % inputs  - L: log-leaders and their estimated covariance structure
    %         - maxit: maximum number of iterations for BFGS algorithm
    %         - lbd_v, lbd_h: regularization parameters
    %
    % outputs - x.h: regularized local regularity obtained with chosen
    %         hyperparameters
    %         - x.v: regularized local power obtained with chosen
    %         hyperparameters
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    % Chosen regularization parameters
    prox.lambda_v = lbd_v;
    prox.lambda_h = lbd_h;
    
    % Minimization of the ROF functional
    [v,h] = PA_PDj(L,prox);
    x.h = h;
    x.v = v;
    x.meth = 'J';
    
end