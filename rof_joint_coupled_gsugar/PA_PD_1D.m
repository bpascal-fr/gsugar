% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize Total Variation denoising functional on linear regression estimate
% of local regularity (signal)
%
% from 
% - A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging, J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)
% and
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246


function [h,crit, gap, t]=PA_PD_1D(L_X,prox)
    
    % inputs  - L_X.h_LR: linear regression estimate of local regularity
    %         - prox.lambda: regularization parameter
    %
    % ouputs  - h: regularized estimate of local regularity
    %         - t: time
    %         - crit: evolution of objective function
    %         - gap: evolution of duality gap
    
    % Implementation B. Pascal, ENS Lyon
    % June 2019
    
    
    % Linear regression local regularity estimate
    h_LR = L_X.h_LR;
    
    
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
    h = zeros(size(h_LR));
    y = opL(h);
    by = y;

    
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
        h = prox_g(h, h_LR,tau);
        
        
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
        crit(it) = 1/2*norm(h-h_LR,'fro')^2 + lambda*sum(abs(p));
        
        % Compute dual criterion
        p = opL_adj_1D(y);
        q = prox_L1(y,lambda);
        dual = 1/2*norm(-p ,'fro')^2 - sum(p.*h_LR) + norm(q,'fro');
        
        % Compute duality gap
        gap(it) = crit(it) + dual;
        gapc = 2*gap(it)/(abs(crit(it)) + abs(dual));
        
        
        
    end
    t = t(1:it);
    crit = crit(1:it);
    gap = gap(1:it);
    
    t = cumsum(t);
end
