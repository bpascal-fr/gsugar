% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to perform Total Variation denoising (image)
%
% from A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging. J. Math. Imag. Vis. 40(1),
% 120-145 (2011)


function [h,crit, gap, t]=PA_PDtv_1D(X,prox)
    
    % inputs  - X.signal_n: observed signal to be denoised
    %         - prox.lambda: regularization parameter
    %
    % ouputs  - h: denoised signal
    %         - t: time
    %         - crit: evolution of objective function
    %         - gap: evolution of duality gap
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    signal_n  = X.signal_n;
    
    %% Fixing Proximal Parameters
    
    gamma = 0.99;
    mu_g=1;
    normD = 1;
    tau = gamma/normD;
    sig = gamma/normD;
    lambda = prox.lambda;
    iter = 1e5;
    eps = 1e-4;
    
    
    %% Initializing variables
    
    
    h= sparse(zeros(size(signal_n)));
    y=sparse(zeros(size(signal_n)));
    by=sparse(zeros(size(signal_n)));
    
    %% Criterion of convergence
    
    crit=zeros(1,iter);
    gap=zeros(1,iter);
    t=zeros(1,iter);
    it = 0;
    gapc = eps+1;
    
    %% Algorithm
    while (gapc > eps)&&(it<iter)
        
        it = it + 1;
        tic
        
        %Save the dual variables
        ys = y;
        
        
        %Update of primal variable
        h = h - tau*opL_adj_1D(by);
        h = prox_g(h, signal_n,tau);
        
        
        %Update of dual variable
        tm = opL_1D(h);
        tily = ys + sig * tm;
        temp = prox_L1(tily/sig, lambda/sig);
        y = tily - sig*temp;
        
        %Update of the descent steps
        theta = (1+2*mu_g*tau)^(-1/2);
        tau = theta*tau;
        sig=sig/theta;
        
        %Update dual auxiliary variable
        by = y + theta*(y - ys);
        
        
        t(it) = toc;
        
        % Compute primal criterion
        p = opL_1D(h);
        crit(it) = 1/2*norm(h-signal_n,'fro')^2 + lambda*sum(abs(p));
        
        % Compute dual criterion
        p = opL_adj_1D(y);
        q = prox_L1(y,lambda);
        dual = 1/2*norm(-p ,'fro')^2 - sum(p.*signal_n) + norm(q,'fro');
        
        % Compute duality gap
        gap(it) = crit(it) + dual;
        gapc = 2*gap(it)/(abs(crit(it)) + abs(dual));
        
        
        
    end
    
    t = t(1:it);
    crit = crit(1:it);
    gap = gap(1:it);
    
    t = cumsum(t);
end
