% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize the coupled functional for fractal texture segmentation (image)
%
% from 
% - A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging, J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)
% and
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246

function [v, h, t, crit, gap]=PA_PDc(L_X,prox)
    
    % inputs  - L_X.leaders: undecimated wavelet leaders
    %         - prox: (lambda, alpha) regularization parameters
    %
    % ouputs  - v: regularized estimate of local power
    %         - h: regularized estimate of local regularity
    %         - t: time
    %         - crit: evolution of objective function
    %         - gap: evolution of duality gap
    %
    % Implementation B. Pascal, ENS Lyon
    % June 2019
    
    %% From leaders to log-leaders
    leaders = L_X.leaders;
    JJ = L_X.JJ;
    L = cell(1,JJ(end));
    for jj=JJ
        L{jj} = log2(leaders{jj});
    end
    
    %% Usefull quantities for prox data fid
    use = usefull(JJ, L);
    
    %% Constant terms for duality gap
    % only depending on leaders
    adj_tv = zeros(size(L{JJ(1)}));
    adj_th = adj_tv;
    for jj=JJ
        adj_tv = adj_tv + L{jj};
        adj_th = adj_th + jj*L{jj};
    end
    
    Aadj_tv = (use.S2*adj_tv - use.S1*adj_th)/use.det;
    Aadj_th = (use.S0*adj_th - use.S1*adj_tv)/use.det;
    
    AAadj_t = cell(1,JJ(end));
    for jj=JJ
        AAadj_t{jj} = Aadj_tv + jj*Aadj_th;
    end
    
    f_adj0 = 0;
    for jj=JJ
        f_adj0 = f_adj0 - 1/2*norm(AAadj_t{jj}-L{jj},'fro')^2;
    end
    
    
    %% Fixing Proximal Paramters
    gamma = 0.99;
    mu_g=use.mu;
    alpha = prox.alpha;
    use.alpha = alpha;
    normD = normD_2D(alpha,size(L{JJ(1)}));
    Lip = normD;
    tau = gamma/Lip;
    sig = gamma/Lip;
    lambda = prox.lambda;
    iter = 1e5;
    eps = 1e-4;
    
    %% Initialization
    h = zeros(size(L{JJ(1)}));
    [g_h, g_v] = opL(h);
    bg_h = g_h;
    bg_v = g_v;
    v = zeros(size(L{JJ(1)}));
    [t_h, t_v] = opL(v);
    bt_h = t_h;
    bt_v = t_v;
    
    crit = zeros(1,iter); gap = crit;
    t = crit;
    
    it = 0;
    gapc = eps + 1;
    
    %% ALGORITHM
    
    while (gapc > eps)&&(it<iter)
        
        it = it + 1;
        
        tic
        %Save the dual variables
        g_hs = g_h;
        g_vs = g_v;
        t_hs = t_h;
        t_vs = t_v;
        
        %Update of primal variable
        h = h - tau*alpha*opL_adj(bg_h,bg_v);
        v = v - tau*opL_adj(bt_h,bt_v);
        [h, v] = prox_hv(h,v,use,tau);
        
        %Update of dual variables
        
        %Dual Variable of h
        [pgn_h, pgn_v] = opL(h);
        g_h = g_h + sig*alpha*pgn_h; %prox argument
        g_v = g_v + sig*alpha*pgn_v; %prox argument
        
        %Dual Variable of s
        [ptn_h, ptn_v] = opL(v);
        t_h = t_h + sig*ptn_h; %prox argument
        t_v = t_v + sig*ptn_v; %prox argument
        
        
        [pgn_h, pgn_v, ptn_h, ptn_v] = prox_L12c(g_h/sig, g_v/sig, t_h/sig, t_v/sig, lambda/sig);
        
        g_h=g_h - sig*pgn_h;
        g_v=g_v - sig*pgn_v;
        
        t_h=t_h - sig*ptn_h;
        t_v=t_v - sig*ptn_v;
        
        %Update of the descent steps
        theta = (1+2*mu_g*tau)^(-1/2);
        tau = theta*tau;
        sig=sig/theta;
        
        
        %Update dual auxiliary variable
        bg_h = g_h + theta*(g_h-g_hs);
        bg_v = g_v + theta*(g_v-g_vs);
        bt_h = t_h + theta*(t_h-t_hs);
        bt_v = t_v + theta*(t_v-t_vs);
        
        t(it) = toc;
        
        
        [c,~,g] = PDc_gap(v, h, t_h, t_v, g_h, g_v, L, adj_tv, adj_th, f_adj0, JJ, prox, use);
        crit(it) = c;
        gap(it) = g;
        gapc = 2*gap(it)/(abs(crit(it)) + abs(gap(it)-crit(it)));
        
        
        
    end
    t = cumsum(t);
    t = t(1:it);
    crit = crit(1:it);
    gap = gap(1:it);
end