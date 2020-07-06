% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize the coupled functional for fractal process segmentation (signal)
%
% from B. Pascal, N. Pustelnik, P. Abry: How Joint Fractal Features Estimation 
% and Texture Segmentation can be cast into a Strongly Convex Optimization
% Problem ?. arxiv:1910.05246
% and A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging. J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)

function [v, h, t, crit, gap]=PA_PDc_1D(L_X,prox)
    
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
    J1=L_X.JJ(1);
    J2=L_X.JJ(end);
    JJ = L_X.JJ;
    
    L = cell(1,JJ(end));
    for jj=J1:J2
        L{jj} = log2(leaders{jj});
    end
    
    %% Usefull quantities for prox data fid
    use = usefull(L_X.JJ, L);
    
    %% Constant terms for duality gap
    % only depending on leaders
    adj_tv = zeros(size(L{J1}));
    adj_th = adj_tv;
    for jj=J1:J2
        adj_tv = adj_tv + L{jj};
        adj_th = adj_th + jj*L{jj};
    end
    
    Aadj_tv = (use.S2*adj_tv - use.S1*adj_th)/use.det;
    Aadj_th = (use.S0*adj_th - use.S1*adj_tv)/use.det;
    
    AAadj_t = cell(1,JJ(end));
    for jj=J1:J2
        AAadj_t{jj} = Aadj_tv + jj*Aadj_th;
    end
    
    f_adj0 = 0;
    for jj=J1:J2
        f_adj0 = f_adj0 - 1/2*norm(AAadj_t{jj}-L{jj},'fro')^2;
    end
    
    
    %% Fixing Proximal Paramters
    gamma = 0.99;
    mu_g = use.mu;
    lambda = prox.lambda;
    alpha = prox.alpha;
    Lip = normD_1D(alpha,size(L{JJ(1)}));
    tau = gamma/Lip;
    sig = gamma/Lip;
    iter = 1e6;
    eps = 1e-4;
    
    %% Initialization
    h = zeros(size(L{J1}));
    gn = alpha*opL_1D(h);
    bgn = gn;
    v = zeros(size(L{J1}));
    tn = opL_1D(v);
    btn = tn;
    
    crit = zeros(1,iter);
    gap = crit;
    t = zeros(1,iter);
    it = 0;
    gap_c = eps + 1;
    
    %% ALGORITHM
    
    if lambda < 0 || lambda < 0
        [v,h] = linear_reg(L,L_X.JJ);
    else
        while (gap_c > eps)&&(it<iter)
            
            it = it + 1;
            
            %% Update variables
            
            tic
            %Save the dual variables
            gn_s = gn;
            tns = tn;
            
            %Update of primal variable
            h = h - tau*alpha*opL_adj_1D(bgn);
            v = v - tau*opL_adj_1D(btn);
            [h, v] = prox_hv(h,v,use,tau);
            
            %Update of dual variables
            
            pgn = alpha*opL_1D(h);
            ptn = opL_1D(v);
            gn = gn + sig*pgn; %prox argument
            tn = tn + sig*ptn; %prox argument
            [pgn, ptn] = prox_L1c(gn/sig, tn/sig, lambda/sig);
            gn = gn - sig*pgn;
            tn = tn - sig*ptn;
            
            
            %Update of the descent steps
            theta = (1+2*mu_g*tau)^(-1/2);
            tau = theta*tau;
            sig=sig/theta;
            
            %Update dual auxiliary variable
            bgn = gn + theta*(gn-gn_s);
            btn = tn + theta*(tn-tns);
            
            t(it) = toc;
            
            %% Compute convergence criteria
            
            [c,~,g] = PDc_gap_1D(v, h, tn, gn, L, adj_tv, adj_th, f_adj0, JJ, prox, use);
            
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