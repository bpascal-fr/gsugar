% Automated selection of the regularization parameters of the 
% coupled functional for fractal texture segmentation using BFGS algorithm 
% and generalized SURE and SUGAR  estimates (image)
%
% from 
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246
% and
% - B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [x,lbd] = bfgs_coupled_gsugar(L,maxit,lbd_in)
    
    % inputs  - L: log-leaders and their estimated covariance structure
    %         - maxit: maximum number of iterations for BFGS algorithm
    %         - lbd_in: initial hyperparameters from which BFGS algorithm starts
    %
    % outputs - x.h: regularized local regularity obtained with optimal
    %         hyperparameters
    %         - x.v: regularized local power obtained with optimal
    %         hyperparameters
    %         - lbd.l: optimal hyperparameter lambda found running BFGS algorithm
    %         - lbd.a: optimal hyperparameter alpha found running BFGS algorithm
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    h_LR = L.h_LR;    % Linear regression estimate of local regularity
    v_LR = L.v_LR;    % Linear regression estimate of local power
    S = L.S;          % Spatial averaging estimate of covariance of log2-leaders
    D = L.D;          % Spatial averaging estimate of variance of log2-leaders
    JJ = L.JJ;        % Range of scales considered
    
    % Compute Finite Difference Monte Carlo parameters
    % sure.eps: finite difference step
    % sure.delta: Monte Carlo unitary Gaussian vector
    sure.eps = 2*max(sqrt(D))/(length(JJ)*numel(h_LR))^.3;
    for jj = JJ
        sure.delta{jj} = randn(size(h_LR));
    end
    
    % Initialization of BFGS quasi-newton
    if nargin == 3
        lbd_li = lbd_in.l;
        lbd_ai = lbd_in.a;
    else
        [Hh, Vh] = opL(h_LR); TV_h = sum(sum((Hh.^2 + Vh.^2).^(1/2)));
        [Hv, Vv] = opL(v_LR); TV_v = sum(sum((Hv.^2 + Vv.^2).^(1/2)));
        lbd_li = length(JJ)*numel(h_LR)*sum(D)/(2*TV_h+2*TV_v);
        lbd_ai = TV_v/TV_h;
    end
    lbd_i = [lbd_li; lbd_ai];
    opts.x0 = lbd_i;
    
    set_init(struct);
    
    % Compute initial Hessian
    [~,sugar] = SURE_SUGAR_coupled_GRANSO(L, lbd_i, sure, S);
    alpha = 0.5; B = zeros(2,2); B(1,1) = abs(alpha*lbd_i(1))/abs(sugar(1)); B(2,2) = abs(alpha*lbd_i(2))/abs(sugar(2));
    opts.H0 = B;
    
    % Define SURE and SUGAR as a function
    SURE_SUGAR = @(lbd) SURE_SUGAR_coupled_GRANSO(L, lbd, sure, S);
    
    % Run BFGS GRANSO algorithm
    opts.print_level = 1;
    opts.quadprog_info_msg = false;
    opts.prescaling_info_msg = false;
    if nargin == 1
        opts.maxit = 20;
    else
        opts.maxit = maxit;
    end
    soln    = granso(2,SURE_SUGAR,@positivityConstraint,[],opts);
    lbd_opt = soln.final.x;
    
    lbd.a = lbd_opt(2);
    lbd.l = lbd_opt(1);
    
    init_PD = get_init;
    x.h = init_PD.h;
    x.v = init_PD.v;
    
    x.meth = 'C';
    
    % Define positivity constraints
    function [ci,ci_grad] = positivityConstraint(lbd)
        % Impose that lbd >= 1e-2 * lbd_i (so that lbd > 0)
        ci = [1e-2 * lbd_i(1) - lbd(1); 1e-2 * lbd_i(2) - lbd(2)];
        ci_grad = [[-1; 0], [0; -1]];
    end
    
    
end