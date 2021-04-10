% Fast Iterative Soft Thresholding Algorithm for the minimization of the
% coupled functional for fractal texture segmentation (image)
%
% from
% - A.Chambolle, C. Dossal: On the convergence of the iterates of ``FISTA",
%  J. Optim. Theory Appl. 166(3), 25 (2015)
% and
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for
% Joint Fractal Feature Estimation and Texture Segmentation,
% (2019) arxiv:1910.05246

function [v, h, t, crit, gap]=FISTAc(L_X,prox)
    
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
    lambda = prox.lambda;
    alpha = prox.alpha;
    normD = sqrt(1+alpha^2)*sqrt(2);
    norminv = use.norminv;
    Lip = normD^2*norminv^2;
    gamman = gamma/Lip;
    iter = 1e5;
    eps = 1e-4;
    
    
    %% Initialization
    
    %Dual variables
    th = zeros(size(L_X.h_LR));
    tv = zeros(size(L_X.h_LR));
    gh = zeros(size(L_X.h_LR));
    gv = zeros(size(L_X.h_LR));
    bth = th;
    btv = tv;
    bgh = gh;
    bgv = gv;
    
    %Primal variables
    v = Aadj_tv;
    h = Aadj_th;
    
    %Intertia FISTA parameter
    fista = 1;
    a = 4;
    
    %Convergnce criteria
    crit = zeros(1,iter);
    gap = crit;
    t = zeros(1,iter);
    it = 0;
    gap_c = eps + 1;
    
    %% ALGORITHM
    
    if lambda < 0 || alpha < 0
        [v,h] = linear_reg(L,L_X.JJ);
    else
        while (gap_c > eps)&&(it<iter)
            
            it = it + 1;
            
            tic
            %Save the dual variables
            ths = th;
            tvs = tv;
            bths = bth;
            btvs = btv;
            ghs = gh;
            gvs = gv;
            bghs = bgh;
            bgvs = bgv;
            
            %Update of dual variables
            [tvh,tvv] = opL(v);
            bth = bth + gamman*tvh;
            btv = btv + gamman*tvv;
            [thh,thv] = opL(h);
            bgh = bgh + gamman*alpha*thh;
            bgv = bgv + gamman*alpha*thv;
            [tvh, tvv, thh, thv] = prox_L12c(bth/gamman,btv/gamman,bgh/gamman,bgv/gamman,lambda/gamman);
            th = bth - gamman*tvh;
            tv = btv - gamman*tvv;
            gh = bgh - gamman*thh;
            gv = bgv - gamman*thv;
            
            %Update of intertia FISTA parameter
            fistas = fista;
            fista = (it+a)/a;
            
            %Update of dual auxiliary variable
            bth = th+(fistas-1)*(th-ths)/fista;
            btv = tv+(fistas-1)*(tv-tvs)/fista;
            bgh = gh+(fistas-1)*(gh-ghs)/fista;
            bgv = gv+(fistas-1)*(gv-gvs)/fista;
            
            %Update of primal variable
            [ov,oh] = op_AtAinvDadj(bth-bths,btv-btvs,alpha*(bgh-bghs),alpha*(bgv-bgvs),use);
            v = v - ov;
            h = h - oh;
            
            t(it) = toc;
            
            
            [c,~,g] = PDc_gap(v, h, th, tv, gh, gv, L, adj_tv, adj_th, f_adj0, JJ, prox, use);
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