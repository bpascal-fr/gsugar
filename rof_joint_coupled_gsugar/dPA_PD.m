% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize Rudin-Osher-Fatemi functional on linear regression estimate
% of local regularity with iterative differentiation w.r.t. the
% regularization parameter (image)
%
% from A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging. J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)
% and B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434



function [h,Eh, dh, Edh, crit, gap, t]=dPA_PD(L_X,prox, sure)

    
    % inputs  - L_X.h_LR: linear regression estimate of local regularity
    %         - prox.lambda: regularization parameter
    %         - sure: delta (Monte Carlo vector) eps (Finite Difference step)
    %
    % ouputs  - h: regularized estimate of local regularity
    %         - Eh: perturbed regularized estimate of local regularity
    %         - dh: gradient of regularized estimate of local regularity
    %         w.r.t. regularization parameter
    %         - Edh: gradient of perturbed regularized estimate of local regularity
    %         w.r.t. regularization parameter
    %         - t: time
    %         - crit: evolution of objective function
    %         - gap: evolution of duality gap
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    % Linear regression local regularity estimate
    h_LR = L_X.h_LR;
    
    % Perturbed local regularity estimate
    leaders = L_X.leaders;
    JJ = L_X.JJ;
    L_per = cell(1,JJ(end));
    for jj=JJ
        L_per{jj} = log2(leaders{jj}) + sure.eps*sure.delta{jj};
    end
    [~, Eh_LR] = linear_reg(L_per, JJ);
    
    %% Fixing Proximal Parameters
    
    gamma = 0.99;
    mu_g=1;
    normD = sqrt(2);
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
        dyh = init_PD.dyh;
        yh = init_PD.yh;
        Edyh = init_PD.Edyh;
        Eyh = init_PD.Eyh;
        dyv = init_PD.dyv;
        yv = init_PD.yv;
        Edyv = init_PD.Edyv;
        Eyv = init_PD.Eyv;
        dbyh = init_PD.dbyh;
        byh = init_PD.byh;
        Edbyh = init_PD.Edbyh;
        Ebyh = init_PD.Ebyh;
        dbyv = init_PD.dbyv;
        byv = init_PD.byv;
        Edbyv = init_PD.Edbyv;
        Ebyv = init_PD.Ebyv;
    else
        dh= sparse(zeros(size(h_LR))); 
        h= sparse(zeros(size(h_LR)));
        Edh= sparse(zeros(size(h_LR)));
        Eh= sparse(zeros(size(h_LR)));
        [tmph,tmpv] = opL(h_LR);
        dyh=sparse(zeros(size(tmph)));
        dyv=sparse(zeros(size(tmpv)));
        yh=sparse(zeros(size(tmph)));
        yv=sparse(zeros(size(tmpv)));
        Edyh=sparse(zeros(size(tmph)));
        Edyv=sparse(zeros(size(tmpv)));
        Eyh=sparse(zeros(size(tmph)));
        Eyv=sparse(zeros(size(tmpv)));
        dbyh=sparse(zeros(size(tmph)));
        dbyv=sparse(zeros(size(tmpv)));
        byh=sparse(zeros(size(tmph)));
        byv=sparse(zeros(size(tmpv)));
        Edbyh=sparse(zeros(size(tmph)));
        Edbyv=sparse(zeros(size(tmpv)));
        Ebyh=sparse(zeros(size(tmph)));
        Ebyv=sparse(zeros(size(tmpv)));
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
        dyhs = dyh;
        dyvs = dyv;
        yhs = yh;
        yvs = yv;
        Edyhs = Edyh;
        Edyvs = Edyv;
        Eyhs = Eyh;
        Eyvs = Eyv;
        
        
        %Update of primal variable
        dh = dh - tau*opL_adj(dbyh,dbyv);
        h = h - tau*opL_adj(byh,byv);
        Edh = Edh - tau*opL_adj(Edbyh,Edbyv);
        Eh = Eh - tau*opL_adj(Ebyh,Ebyv);
        
        
        dh = prox_g(dh, zeros(size(h_LR)),tau);
        h = prox_g(h, h_LR,tau);
        Edh = prox_g(Edh, zeros(size(h_LR)),tau);
        Eh = prox_g(Eh, Eh_LR,tau);
        
        
        %Update of dual variable
        [dth, dtv] = opL(dh);
        [th, tv] = opL(h);
        [Edth, Edtv] = opL(Edh);
        [Eth, Etv] = opL(Eh);
        
        dtilyh = dyhs + sig * dth;
        dtilyv = dyvs + sig * dtv;
        tilyh = yhs + sig * th;
        tilyv = yvs + sig * tv;
        Edtilyh = Edyhs + sig * Edth;
        Edtilyv = Edyvs + sig * Edtv;
        Etilyh = Eyhs + sig * Eth;
        Etilyv = Eyvs + sig * Etv;
        
        [dtemph, dtempv] = dprox_L12_lambda(tilyh/sig, tilyv/sig, lambda/sig);
        [d1dtemph, d1dtempv] = dprox_L12(tilyh/sig, tilyv/sig,dtilyh, dtilyv, lambda/sig);
        [temph, tempv] = prox_L12(tilyh/sig, tilyv/sig, lambda/sig);
        [Edtemph, Edtempv] = dprox_L12_lambda(Etilyh/sig, Etilyv/sig, lambda/sig);
        [d1Edtemph, d1Edtempv] = dprox_L12(Etilyh/sig, Etilyv/sig,Edtilyh, Edtilyv, lambda/sig);
        [Etemph, Etempv] = prox_L12(Etilyh/sig, Etilyv/sig, lambda/sig);
        
        dyh = dtilyh - dtemph - d1dtemph;
        dyv = dtilyv - dtempv - d1dtempv;
        yh = tilyh - sig*temph;
        yv = tilyv - sig*tempv;
        Edyh = Edtilyh - Edtemph - d1Edtemph;
        Edyv = Edtilyv - Edtempv - d1Edtempv;
        Eyh = Etilyh - sig*Etemph;
        Eyv = Etilyv - sig*Etempv;
        
        %Update of the descent steps
        theta = (1+2*mu_g*tau)^(-1/2);
        tau = theta*tau;
        sig=sig/theta;
        
        %Update dual auxiliary variable
        dbyh = dyh + theta*(dyh - dyhs);
        dbyv = dyv + theta*(dyv - dyvs);
        byh = yh + theta*(yh - yhs);
        byv = yv + theta*(yv - yvs);
        Edbyh = Edyh + theta*(Edyh - Edyhs);
        Edbyv = Edyv + theta*(Edyv - Edyvs);
        Ebyh = Eyh + theta*(Eyh - Eyhs);
        Ebyv = Eyv + theta*(Eyv - Eyvs);
        
        
        
        t(it) = toc;
        
        % Compute primal criterion
        [ph,pv] = opL(h);
        crit(it) = 1/2*norm(h-h_LR,'fro')^2 + lambda*sum(sum((ph.^2+pv.^2).^(1/2)));
        
        % Compute dual criterion
        p = opL_adj(yh,yv);
        [p1, p2] = prox_L12(yh,yv,lambda);
        dual = 1/2*norm(-p ,'fro')^2 - sum(p(:).*h_LR(:)) + norm([p1,p2],'fro');
        
        % Compute duality gap
        gap(it) = crit(it) + dual;
        gapc = 2*gap(it)/(abs(crit(it)) + abs(dual));
        
        
        
    end
    
    init_PD.dh = dh;
    init_PD.h = h;
    init_PD.Edh = Edh;
    init_PD.Eh = Eh;
    init_PD.dyh = dyh;
    init_PD.yh = yh;
    init_PD.Edyh = Edyh;
    init_PD.Eyh = Eyh;
    init_PD.dyv = dyv;
    init_PD.yv = yv;
    init_PD.Edyv = Edyv;
    init_PD.Eyv = Eyv;
    init_PD.dbyh = dbyh;
    init_PD.byh = byh;
    init_PD.Edbyh = Edbyh;
    init_PD.Ebyh = Ebyh;
    init_PD.dbyv = dbyv;
    init_PD.byv = byv;
    init_PD.Edbyv = Edbyv;
    init_PD.Ebyv = Ebyv;
    
    set_init(init_PD);
    
    t = t(1:it);
    crit = crit(1:it);
    gap = gap(1:it);
    
    t = cumsum(t);
end
