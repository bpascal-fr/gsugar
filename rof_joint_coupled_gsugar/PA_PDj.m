% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize the joint functional for fractal texture segmentation (image)
%
% from B. Pascal, N. Pustelnik, P. Abry: How Joint Fractal Features Estimation 
% and Texture Segmentation can be cast into a Strongly Convex Optimization
% Problem ?. arxiv:1910.05246
% and A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging. J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)

function [v, h, t, crit, gap]=PA_PDj(L_X,prox)
    
    % inputs  - L_X.leaders: undecimated wavelet leaders
    %         - prox: (lambda_v, lambda_h) regularization parameters
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
    JJ=L_X.JJ;
    L = cell(1,JJ(end));
    for jj=JJ
        L{jj} = log2(leaders{jj});
    end
    
    %% Usefull quantities for prox data fid
    use = usefull(L_X.JJ, L);
    
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
    mu_g = use.mu;
    normD = sqrt(2);
    Lip = normD;
    tau = gamma/Lip;
    sig = gamma/Lip;
    lambda_h = prox.lambda_h;
    lambda_v = prox.lambda_v;
    iter = 1e5;
    eps = 1e-4;
    
    %% Initialization
    h = zeros(size(L{JJ(1)}));
    [gn_h, gn_v] = opL(h);
    bgn_h = gn_h;
    bgn_v = gn_v;
    v = zeros(size(L{JJ(1)}));
    [tn_h, tn_v] = opL(v);
    btn_h = tn_h;
    btn_v = tn_v;
    
    crit = zeros(1,iter);
    gap = crit;
    t = zeros(1,iter);
    it = 0;
    gap_c = eps + 1;
    
    %% ALGORITHM
    
    if lambda_v < 0 || lambda_h < 0
        [v,h] = linear_reg(L,L_X.JJ);
    else
        while (gap_c > eps)&&(it<iter)
            
            it = it + 1;
            
            %% Update variables
            
            tic
            %Save the dual variables
            gn_hs = gn_h;
            gn_vs = gn_v;
            tn_hs = tn_h;
            tn_vs = tn_v;
            
            %Update of primal variable
            h = h - tau*opL_adj(bgn_h,bgn_v);
            v = v - tau*opL_adj(btn_h,btn_v);
            [h, v] = prox_hv(h,v,use,tau);
            
            %Update of dual variables
            
            %Dual Variable of h
            [pgn_h, pgn_v] = opL(h);
            gn_h = gn_h + sig*pgn_h; %prox argument
            gn_v = gn_v + sig*pgn_v; %prox argument
            [pgn_h, pgn_v] = prox_L12(gn_h/sig, gn_v/sig, lambda_h/sig);
            gn_h=gn_h - sig*pgn_h;
            gn_v=gn_v - sig*pgn_v;
            
            %Dual Variable of s
            [ptn_h, ptn_v] = opL(v);
            tn_h = tn_h + sig*ptn_h; %prox argument
            tn_v = tn_v + sig*ptn_v; %prox argument
            [ptn_h, ptn_v] = prox_L12(tn_h/sig, tn_v/sig, lambda_v/sig);
            tn_h=tn_h - sig*ptn_h;
            tn_v=tn_v - sig*ptn_v;
            
            %Update of the descent steps
            theta = (1+2*mu_g*tau)^(-1/2);
            tau = theta*tau;
            sig=sig/theta;
            
            %Update dual auxiliary variable
            bgn_h = gn_h + theta*(gn_h-gn_hs);
            bgn_v = gn_v + theta*(gn_v-gn_vs);
            btn_h = tn_h + theta*(tn_h-tn_hs);
            btn_v = tn_v + theta*(tn_v-tn_vs);
            
            t(it) = toc;
            
            %% Compute convergence criteria
            
            [c,~,g] = PDj_gap(v, h, tn_h, tn_v, gn_h, gn_v, L, adj_tv, adj_th, f_adj0, JJ, prox, use);
            
            crit(it) = c;
            gap(it) = g;
            gap_c = 2*gap(it)/(abs(crit(it)) + abs(gap(it)-crit(it)));
            
            
            
        end
        
        t = cumsum(t);
        t = t(1:it);
        crit = crit(1:it);
        gap = gap(1:it);
        
    end
end