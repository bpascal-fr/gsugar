% Automated selection of the regularization parameter for Total Variation 
% denoising using BFGS algorithm and generalized SURE and SUGAR estimates 
% (signal)
%
% from B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [x,lbd] = bfgs_tv_gsugar_1D(X,maxit,lbd_in)
    
    % inputs  - X.signal_n: observed signal to be denoised and its covariance
    %         structure
    %         - maxit: maximum number of iterations for BFGS algorithm
    %         - lbd_in: initial hyperparameter from which BFGS algorithm starts
    %
    % outputs - x.signal: denoised signal obtained with optimal hyperparameter
    %         - lbd: optimal hyperparameter found running BFGS algorithm
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    S   = X.S;
    D   = X.D;
    signal   = X.signal_n;
    
    % Compute Finite Difference Monte Carlo parameters
    % sure.eps: finite difference step
    % sure.delta: Monte Carlo unitary Gaussian vector
    sure.eps = 2*sqrt(D)/(numel(signal))^.3;
    sure.delta = randn(size(signal));

    
    % Initialization of BFGS quasi-newton
    if nargin == 3
        lbd_i = lbd_in.l;
    else
        Dsignal = opL_1D(signal); TV_signal = sum(abs(Dsignal));
        lbd_i = numel(signal)*D/(2*TV_signal);
    end
    opts.x0 = lbd_i;
    
    set_init(struct);
    
    % Compute initial Hessian
    [~,sugar] = SURE_SUGAR_tv_GRANSO_1D(X, lbd_i, sure, S);
    alpha = 0.5; B = abs(alpha*lbd_i(1))/abs(sugar);
    opts.H0 = B;
    
    % Define SURE and SUGAR as a function
    SURE_SUGAR = @(lbd) SURE_SUGAR_tv_GRANSO_1D(X, lbd, sure, S);
    
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
    x.signal = init_PD.h;
    
    x.meth = 'TV';
    
    % Define positivity constraints
    function [ci,ci_grad] = positivityConstraint(lbd)
        % Impose that lbd >= 1e-2 * lbd_i (so that lbd > 0)
        ci = 1e-2 * lbd_i - lbd;
        ci_grad = -1;
    end
    
    
end