% Automated selection of the regularization parameter of the 
% Rudin-Osher-Fatemi functional applied on the linear regression estimate 
% of local regularity using BFGS algorithm and generalized SURE and SUGAR 
% estimates (signal)
%
% from B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [x,lbd] = bfgs_rof_gsugar_1D(L,maxit,lbd_in)
    
    % inputs  - L: log-leaders and their estimated covariance structure
    %         - maxit: maximum number of iterations for BFGS algorithm
    %         - lbd_in: initial hyperparameter from which BFGS algorithm starts
    %
    % outputs - x.h: regularized local regularity obtained with optimal
    %         hyperparameter
    %         - lbd: optimal hyperparameter found running BFGS algorithm
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    h_LR = L.h_LR;    % Linear regression estimate of local regularity
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
        lbd_i = lbd_in.l;
    else
        Lh = opL_1D(h_LR); TV_h = sum(abs(Lh));
        lbd_i = numel(h_LR)*sum(D)/(2*TV_h);
    end
    opts.x0 = lbd_i;
    
    set_init(struct);
    
    % Compute initial Hessian
    [~,sugar] = SURE_SUGAR_rof_GRANSO_1D(L, lbd_i, sure, S);
    alpha = 0.5; B = abs(alpha*lbd_i)/abs(sugar);
    opts.H0 = B;
    
    % Define SURE and SUGAR as a function
    SURE_SUGAR = @(lbd) SURE_SUGAR_rof_GRANSO_1D(L, lbd, sure, S);
    
    % Run BFGS GRANSO algorithm
    opts.print_level = 1;
    opts.quadprog_info_msg = false;
    opts.prescaling_info_msg = false;
    if nargin == 1
        opts.maxit = 20;
    else
        opts.maxit = maxit;
    end
    soln    = granso(1,SURE_SUGAR,@positivityConstraint,[],opts);
    lbd_opt = soln.final.x;
    
    lbd.l = lbd_opt;
    
    init_PD = get_init;
    x.h = init_PD.h;
    
    x.meth = 'ROF';
    
    % Define positivity constraints
    function [ci,ci_grad] = positivityConstraint(lbd)
        % Impose that lbd >= 1e-2 * lbd_i (so that lbd > 0)
        ci = 1e-2 * lbd_i - lbd;
        ci_grad = -1;
    end
    
    
end