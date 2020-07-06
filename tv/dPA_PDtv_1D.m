% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to perform Total Variation denoising with iterative differentiation w.r.t. the
% regularization parameter (signal)
%
% from A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging. J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)
% and B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [h,Eh, dh, Edh, crit, gap, t]=dPA_PDtv_1D(X,prox,sure)
    
    % inputs  - X.signal_n: observed signal to be denoised
    %         - prox.lambda: regularization parameter
    %         - sure: delta (Monte Carlo vector) eps (Finite Difference step)
    %
    % ouputs  - h: denoised signal
    %         - Eh: perturbed denoised signal
    %         - dh: gradient of denoised signal
    %         w.r.t. regularization parameter
    %         - Edh: gradient of perturbed denoised signal
    %         w.r.t. regularization parameter
    %         - t: time
    %         - crit: evolution of objective function
    %         - gap: evolution of duality gap
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    signal_n  = X.signal_n;
    Esignal_n = signal_n + sure.eps*sure.delta;
    
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
    
    init_PD = get_init;
    
    if isfield(init_PD,'h')
        dh = init_PD.dh;
        h = init_PD.h;
        Edh = init_PD.Edh;
        Eh = init_PD.Eh;
        dy = init_PD.dy;
        y = init_PD.y;
        Edy = init_PD.Edy;
        Ey = init_PD.Ey;
        dby = init_PD.dby;
        by = init_PD.by;
        Edby = init_PD.Edby;
        Eby = init_PD.Eby;
    else
        dh= sparse(zeros(size(signal_n)));
        h= sparse(zeros(size(signal_n)));
        Edh= sparse(zeros(size(signal_n)));
        Eh= sparse(zeros(size(signal_n)));
        dy=sparse(zeros(size(signal_n)));
        y=sparse(zeros(size(signal_n)));
        Edy=sparse(zeros(size(signal_n)));
        Ey=sparse(zeros(size(signal_n)));
        dby=sparse(zeros(size(signal_n)));
        by=sparse(zeros(size(signal_n)));
        Edby=sparse(zeros(size(signal_n)));
        Eby=sparse(zeros(size(signal_n)));
    end
    
    
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
        dys = dy;
        ys = y;
        Edys = Edy;
        Eys = Ey;
        
        
        %Update of primal variable
        dh = dh - tau*opL_adj_1D(dby);
        h = h - tau*opL_adj_1D(by);
        Edh = Edh - tau*opL_adj_1D(Edby);
        Eh = Eh - tau*opL_adj_1D(Eby);
        
        
        dh = prox_g(dh, zeros(size(signal_n)),tau);
        h = prox_g(h, signal_n,tau);
        Edh = prox_g(Edh, zeros(size(signal_n)),tau);
        Eh = prox_g(Eh, Esignal_n,tau);
        
        
        %Update of dual variable
        dtm = opL_1D(dh);
        tm = opL_1D(h);
        Edtm = opL_1D(Edh);
        Etm = opL_1D(Eh);
        
        dtily = dys + sig * dtm;
        tily = ys + sig * tm;
        Edtily = Edys + sig * Edtm;
        Etily = Eys + sig * Etm;
        
        dtemp = dprox_L1_lambda(tily/sig,lambda/sig);
        d1dtemp = dprox_L1(tily/sig, dtily, lambda/sig);
        temp = prox_L1(tily/sig, lambda/sig);
        Edtemp = dprox_L1_lambda(Etily/sig, lambda/sig);
        d1Edtemp = dprox_L1(Etily/sig, Edtily, lambda/sig);
        Etemp = prox_L1(Etily/sig, lambda/sig);
        
        dy = dtily - dtemp - d1dtemp;
        y = tily - sig*temp;
        Edy = Edtily - Edtemp - d1Edtemp;
        Ey = Etily - sig*Etemp;
        
        %Update of the descent steps
        theta = (1+2*mu_g*tau)^(-1/2);
        tau = theta*tau;
        sig=sig/theta;
        
        %Update dual auxiliary variable
        dby = dy + theta*(dy - dys);
        by = y + theta*(y - ys);
        Edby = Edy + theta*(Edy - Edys);
        Eby = Ey + theta*(Ey - Eys);
        
        
        
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
    
    init_PD.dh = dh;
    init_PD.h = h;
    init_PD.Edh = Edh;
    init_PD.Eh = Eh;
    init_PD.dy = dy;
    init_PD.y = y;
    init_PD.Edy = Edy;
    init_PD.Ey = Ey;
    init_PD.dby = dby;
    init_PD.by = by;
    init_PD.Edby = Edby;
    init_PD.Eby = Eby;
    
    set_init(init_PD);
    
    t = t(1:it);
    crit = crit(1:it);
    gap = gap(1:it);
    
    t = cumsum(t);
end
