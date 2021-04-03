% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize the joint functional for fractal texture segmentation with 
% iterative differentiation w.r.t. the regularization parameters (image)
%
% from 
% - A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging, J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)
% and
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246
% and
% - B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434


function [v, h, Ev, Eh, dv_lv, dv_lh, dh_lv, dh_lh, Edv_lv, Edv_lh, Edh_lv, Edh_lh, crit, gap, t]=dPA_PDj(L_X,prox,sure)
    
    % inputs  - L_X.leaders: undecimated wavelet leaders
    %         - prox: (lambda_v, lambda_h) regularization parameters
    %         - sure: delta (Monte Carlo vector) eps (Finite Difference step)
    %
    % ouputs  - v: regularized estimate of local power
    %         - h: regularized estimate of local regularity
    %         - Ev: perturbed regularized estimate of local power
    %         - Eh: perturbed regularized estimate of local regularity
    %         - dv_lv: gradient of regularized estimate of local power
    %         w.r.t. regularization parameter lambda_v
    %         - dv_lh: gradient of regularized estimate of local power
    %         w.r.t. regularization parameter lambda_h
    %         - dh_lv: gradient of regularized estimate of local regularity
    %         w.r.t. regularization parameter lambda_v
    %         - dh_lh: gradient of regularized estimate of local regularity
    %         w.r.t. regularization parameter lambda_h
    %         - Edv_lv: gradient of perturbed regularized estimate of local power
    %         w.r.t. regularization parameter lambda_v
    %         - Edv_lh: gradient of perturbed regularized estimate of local power
    %         w.r.t. regularization parameter lambda_h
    %         - Edh_lv: gradient of perturbed regularized estimate of local regularity
    %         w.r.t. regularization parameter lambda_v
    %         - Edh_lh: gradient of perturbed regularized estimate of local regularity
    %         w.r.t. regularization parameter lambda_h
    %         - t: time
    %         - crit: evolution of objective function
    %         - gap: evolution of duality gap
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    
    %% From leaders to log-leaders
    
    leaders = L_X.leaders;
    JJ = L_X.JJ;
    L_0 = cell(1,JJ(end));
    L = cell(1,JJ(end));
    L_per = cell(1,JJ(end));
    for jj=JJ
        L_0{jj} = zeros(size(leaders{jj}));
        L{jj} = log2(leaders{jj});
        L_per{jj} = log2(leaders{jj}) + sure.eps*sure.delta{jj};
    end
    
    %% Usefull quantities for prox data fidelity
    
    use_0   = usefull(JJ, L_0);
    use     = usefull(JJ, L);
    use_per = usefull(JJ, L_per);
    
    
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
    lambda_v = prox.lambda_v;
    lambda_h = prox.lambda_h;
    iter = 1e5;
    eps = 1e-4;
    
    %% Initialization
    
    init_PD = get_init;
    
    if isfield(init_PD,'h')
        
        % Local regularity
        dh_lv = init_PD.dh_lv;
        dh_lh = init_PD.dh_lh;
        h = init_PD.h;
        Edh_lv = init_PD.Edh_lv;
        Edh_lh = init_PD.Edh_lh;
        Eh = init_PD.Eh;
        
        dgn_h_lv = init_PD.dgn_h_lv;
        dgn_v_lv = init_PD.dgn_v_lv;
        dgn_h_lh = init_PD.dgn_h_lh;
        dgn_v_lh = init_PD.dgn_v_lh;
        gn_h = init_PD.gn_h;
        gn_v = init_PD.gn_v;
        Edgn_h_lv = init_PD.Edgn_h_lv;
        Edgn_v_lv = init_PD.Edgn_v_lv;
        Edgn_h_lh = init_PD.Edgn_h_lh;
        Edgn_v_lh = init_PD.Edgn_v_lh;
        Egn_h = init_PD.Egn_h;
        Egn_v = init_PD.Egn_v;
        
        
        dbgn_h_lv = init_PD.dbgn_h_lv;
        dbgn_v_lv = init_PD.dbgn_v_lv;
        dbgn_h_lh = init_PD.dbgn_h_lh;
        dbgn_v_lh = init_PD.dbgn_v_lh;
        bgn_h = init_PD.bgn_h;
        bgn_v = init_PD.bgn_v;
        Edbgn_h_lv = init_PD.Edbgn_h_lv;
        Edbgn_v_lv = init_PD.Edbgn_v_lv;
        Edbgn_h_lh = init_PD.Edbgn_h_lh;
        Edbgn_v_lh = init_PD.Edbgn_v_lh;
        Ebgn_h = init_PD.Ebgn_h;
        Ebgn_v = init_PD.Ebgn_v;
        
        
        
        % Local variance
        dv_lv = init_PD.dv_lv;
        dv_lh = init_PD.dv_lh;
        v = init_PD.v;
        Edv_lv = init_PD.Edv_lv;
        Edv_lh = init_PD.Edv_lh;
        Ev = init_PD.Ev;
        
        dtn_h_lv = init_PD.dtn_h_lv;
        dtn_v_lv = init_PD.dtn_v_lv;
        dtn_h_lh = init_PD.dtn_h_lh;
        dtn_v_lh = init_PD.dtn_v_lh;
        tn_h = init_PD.tn_h;
        tn_v = init_PD.tn_v;
        Edtn_h_lv = init_PD.Edtn_h_lv;
        Edtn_v_lv = init_PD.Edtn_v_lv;
        Edtn_h_lh = init_PD.Edtn_h_lh;
        Edtn_v_lh = init_PD.Edtn_v_lh;
        Etn_h = init_PD.Etn_h;
        Etn_v = init_PD.Etn_v;
        
        
        dbtn_h_lv = init_PD.dbtn_h_lv;
        dbtn_v_lv = init_PD.dbtn_v_lv;
        dbtn_h_lh = init_PD.dbtn_h_lh;
        dbtn_v_lh = init_PD.dbtn_v_lh;
        btn_h = init_PD.btn_h;
        btn_v = init_PD.btn_v;
        Edbtn_h_lv = init_PD.Edbtn_h_lv;
        Edbtn_v_lv = init_PD.Edbtn_v_lv;
        Edbtn_h_lh = init_PD.Edbtn_h_lh;
        Edbtn_v_lh = init_PD.Edbtn_v_lh;
        Ebtn_h = init_PD.Ebtn_h;
        Ebtn_v = init_PD.Ebtn_v;
        
    else
        
        % Local regularity
        dh_lv = zeros(size(L{JJ(1)}));
        dh_lh = zeros(size(L{JJ(1)}));
        h = zeros(size(L{JJ(1)}));
        Edh_lv = zeros(size(L{JJ(1)}));
        Edh_lh = zeros(size(L{JJ(1)}));
        Eh = zeros(size(L{JJ(1)}));
        
        [dgn_h_lv, dgn_v_lv] = opL(dh_lv);
        [dgn_h_lh, dgn_v_lh] = opL(dh_lh);
        [gn_h, gn_v] = opL(h);
        [Edgn_h_lv, Edgn_v_lv] = opL(Edh_lv);
        [Edgn_h_lh, Edgn_v_lh] = opL(Edh_lh);
        [Egn_h, Egn_v] = opL(Eh);
        
        dbgn_h_lv = dgn_h_lv;
        dbgn_h_lh = dgn_h_lh;
        bgn_h = gn_h;
        Edbgn_h_lv = Edgn_h_lv;
        Edbgn_h_lh = Edgn_h_lh;
        Ebgn_h = Egn_h;
        
        dbgn_v_lv = dgn_v_lv;
        dbgn_v_lh = dgn_v_lh;
        bgn_v = gn_v;
        Edbgn_v_lv = Edgn_v_lv;
        Edbgn_v_lh = Edgn_v_lh;
        Ebgn_v = Egn_v;
        
        % Local variance
        dv_lv = zeros(size(L{JJ(1)}));
        dv_lh = zeros(size(L{JJ(1)}));
        v = zeros(size(L{JJ(1)}));
        Edv_lv = zeros(size(L{JJ(1)}));
        Edv_lh = zeros(size(L{JJ(1)}));
        Ev = zeros(size(L{JJ(1)}));
        
        [dtn_h_lv, dtn_v_lv] = opL(dv_lv);
        [dtn_h_lh, dtn_v_lh] = opL(dv_lh);
        [tn_h, tn_v] = opL(v);
        [Edtn_h_lv, Edtn_v_lv] = opL(Edv_lv);
        [Edtn_h_lh, Edtn_v_lh] = opL(Edv_lh);
        [Etn_h, Etn_v] = opL(Ev);
        
        dbtn_h_lv = dtn_h_lv;
        dbtn_h_lh = dtn_h_lh;
        btn_h = tn_h;
        Edbtn_h_lv = Edtn_h_lv;
        Edbtn_h_lh = Edtn_h_lh;
        Ebtn_h = Etn_h;
        
        dbtn_v_lv = dtn_v_lv;
        dbtn_v_lh = dtn_v_lh;
        btn_v = tn_v;
        Edbtn_v_lv = Edtn_v_lv;
        Edbtn_v_lh = Edtn_v_lh;
        Ebtn_v = Etn_v;
        
    end
    crit = zeros(1,iter);
    gap = crit;
    t = zeros(1,iter);
    it = 0;
    gap_c = eps + 1;
    
    %% ALGORITHM
    
    if lambda_v < 0 || lambda_h < 0
        [v,h] = linear_reg(L,L_X.JJ);
        [Ev,Eh] = linear_reg(L_per,L_X.JJ);
    else
        while (gap_c > eps)&&(it<iter)
            
            
            it = it + 1;
            
            %% Update variables
            
            tic
            %Save the dual variables
            dgn_hs_lv = dgn_h_lv;
            dgn_hs_lh = dgn_h_lh;
            gn_hs = gn_h;
            Edgn_hs_lv = Edgn_h_lv;
            Edgn_hs_lh = Edgn_h_lh;
            Egn_hs = Egn_h;
            
            dgn_vs_lv = dgn_v_lv;
            dgn_vs_lh = dgn_v_lh;
            gn_vs = gn_v;
            Edgn_vs_lv = Edgn_v_lv;
            Edgn_vs_lh = Edgn_v_lh;
            Egn_vs = Egn_v;
            
            dtn_hs_lv = dtn_h_lv;
            dtn_hs_lh = dtn_h_lh;
            tn_hs = tn_h;
            Edtn_hs_lv = Edtn_h_lv;
            Edtn_hs_lh = Edtn_h_lh;
            Etn_hs = Etn_h;
            
            dtn_vs_lv = dtn_v_lv;
            dtn_vs_lh = dtn_v_lh;
            tn_vs = tn_v;
            Edtn_vs_lv = Edtn_v_lv;
            Edtn_vs_lh = Edtn_v_lh;
            Etn_vs = Etn_v;
            
            %Update of primal variable
            dh_lv = dh_lv - tau*opL_adj(dbgn_h_lv,dbgn_v_lv);
            dh_lh = dh_lh - tau*opL_adj(dbgn_h_lh,dbgn_v_lh);
            h = h - tau*opL_adj(bgn_h,bgn_v);
            Edh_lv = Edh_lv - tau*opL_adj(Edbgn_h_lv,Edbgn_v_lv);
            Edh_lh = Edh_lh - tau*opL_adj(Edbgn_h_lh,Edbgn_v_lh);
            Eh = Eh - tau*opL_adj(Ebgn_h,Ebgn_v);
            
            dv_lv = dv_lv - tau*opL_adj(dbtn_h_lv,dbtn_v_lv);
            dv_lh = dv_lh - tau*opL_adj(dbtn_h_lh,dbtn_v_lh);
            v = v - tau*opL_adj(btn_h,btn_v);
            Edv_lv = Edv_lv - tau*opL_adj(Edbtn_h_lv,Edbtn_v_lv);
            Edv_lh = Edv_lh - tau*opL_adj(Edbtn_h_lh,Edbtn_v_lh);
            Ev = Ev - tau*opL_adj(Ebtn_h,Ebtn_v);
            
            [dh_lv, dv_lv] = prox_hv(dh_lv,dv_lv,use_0,tau);
            [dh_lh, dv_lh] = prox_hv(dh_lh,dv_lh,use_0,tau);
            [h, v] = prox_hv(h,v,use,tau);
            [Edh_lv, Edv_lv] = prox_hv(Edh_lv,Edv_lv,use_0,tau);
            [Edh_lh, Edv_lh] = prox_hv(Edh_lh,Edv_lh,use_0,tau);
            [Eh, Ev] = prox_hv(Eh,Ev,use_per,tau);
            
            %Update of dual variables
            
            %Dual Variable of h
            [dpgn_h_lv, dpgn_v_lv] = opL(dh_lv);
            [dpgn_h_lh, dpgn_v_lh] = opL(dh_lh);
            [pgn_h, pgn_v] = opL(h);
            [Edpgn_h_lv, Edpgn_v_lv] = opL(Edh_lv);
            [Edpgn_h_lh, Edpgn_v_lh] = opL(Edh_lh);
            [Epgn_h, Epgn_v] = opL(Eh);
            
            dgn_h_lv = dgn_h_lv + sig*dpgn_h_lv; %prox argument
            dgn_h_lh = dgn_h_lh + sig*dpgn_h_lh; %prox argument
            gn_h = gn_h + sig*pgn_h; %prox argument
            Edgn_h_lv = Edgn_h_lv + sig*Edpgn_h_lv; %prox argument
            Edgn_h_lh = Edgn_h_lh + sig*Edpgn_h_lh; %prox argument
            Egn_h = Egn_h + sig*Epgn_h; %prox argument
            
            dgn_v_lv = dgn_v_lv + sig*dpgn_v_lv; %prox argument
            dgn_v_lh = dgn_v_lh + sig*dpgn_v_lh; %prox argument
            gn_v = gn_v + sig*pgn_v; %prox argument
            Edgn_v_lv = Edgn_v_lv + sig*Edpgn_v_lv; %prox argument
            Edgn_v_lh = Edgn_v_lh + sig*Edpgn_v_lh; %prox argument
            Egn_v = Egn_v + sig*Epgn_v; %prox argument
            
            [d1dpgn_h_lv, d1dpgn_v_lv] = dprox_L12(gn_h/sig, gn_v/sig,dgn_h_lv, dgn_v_lv, lambda_h/sig);
            [d1dpgn_h_lh, d1dpgn_v_lh] = dprox_L12(gn_h/sig, gn_v/sig,dgn_h_lh, dgn_v_lh, lambda_h/sig);
            %[dpgn_h_lv, dpgn_v_lv] = dprox_L12_lambda(gn_h/sig, gn_v/sig,dgn_h_lv, dgn_v_lv, lambda_h/sig);
            [dpgn_h_lh, dpgn_v_lh] = dprox_L12_lambda(gn_h/sig, gn_v/sig, lambda_h/sig);
            [pgn_h, pgn_v] = prox_L12(gn_h/sig, gn_v/sig, lambda_h/sig);
            [d1Edpgn_h_lv, d1Edpgn_v_lv] = dprox_L12(Egn_h/sig, Egn_v/sig,Edgn_h_lv, Edgn_v_lv, lambda_h/sig);
            [d1Edpgn_h_lh, d1Edpgn_v_lh] = dprox_L12(Egn_h/sig, Egn_v/sig,Edgn_h_lh, Edgn_v_lh, lambda_h/sig);
            %[Edpgn_h_lv, Edpgn_v_lv] = dprox_L12_lambda(Egn_h/sig, Egn_v/sig,Edgn_h_lv, Edgn_v_lv, lambda_h/sig);
            [Edpgn_h_lh, Edpgn_v_lh] = dprox_L12_lambda(Egn_h/sig, Egn_v/sig, lambda_h/sig);
            [Epgn_h, Epgn_v] = prox_L12(Egn_h/sig, Egn_v/sig, lambda_h/sig);
            
            dgn_h_lv=dgn_h_lv - d1dpgn_h_lv;
            dgn_h_lh=dgn_h_lh - dpgn_h_lh - d1dpgn_h_lh;
            gn_h=gn_h - sig*pgn_h;
            Edgn_h_lv=Edgn_h_lv - d1Edpgn_h_lv;
            Edgn_h_lh=Edgn_h_lh - Edpgn_h_lh - d1Edpgn_h_lh;
            Egn_h=Egn_h - sig*Epgn_h;
            
            dgn_v_lv=dgn_v_lv - d1dpgn_v_lv;
            dgn_v_lh=dgn_v_lh - dpgn_v_lh - d1dpgn_v_lh;
            gn_v=gn_v - sig*pgn_v;
            Edgn_v_lv=Edgn_v_lv - d1Edpgn_v_lv;
            Edgn_v_lh=Edgn_v_lh - Edpgn_v_lh - d1Edpgn_v_lh;
            Egn_v=Egn_v - sig*Epgn_v;
            
            %Dual Variable of v
            [dptn_h_lv, dptn_v_lv] = opL(dv_lv);
            [dptn_h_lh, dptn_v_lh] = opL(dv_lh);
            [ptn_h, ptn_v] = opL(v);
            [Edptn_h_lv, Edptn_v_lv] = opL(Edv_lv);
            [Edptn_h_lh, Edptn_v_lh] = opL(Edv_lh);
            [Eptn_h, Eptn_v] = opL(Ev);
            
            dtn_h_lv = dtn_h_lv + sig*dptn_h_lv; %prox argument
            dtn_h_lh = dtn_h_lh + sig*dptn_h_lh; %prox argument
            tn_h = tn_h + sig*ptn_h; %prox argument
            Edtn_h_lv = Edtn_h_lv + sig*Edptn_h_lv; %prox argument
            Edtn_h_lh = Edtn_h_lh + sig*Edptn_h_lh; %prox argument
            Etn_h = Etn_h + sig*Eptn_h; %prox argument
            
            dtn_v_lv = dtn_v_lv + sig*dptn_v_lv; %prox argument
            dtn_v_lh = dtn_v_lh + sig*dptn_v_lh; %prox argument
            tn_v = tn_v + sig*ptn_v; %prox argument
            Edtn_v_lv = Edtn_v_lv + sig*Edptn_v_lv; %prox argument
            Edtn_v_lh = Edtn_v_lh + sig*Edptn_v_lh; %prox argument
            Etn_v = Etn_v + sig*Eptn_v; %prox argument
            
            [dptn_h_lv, dptn_v_lv] = dprox_L12_lambda(tn_h/sig, tn_v/sig,lambda_v/sig);
            %[dptn_h_lh, dptn_v_lh] = dprox_L12_lambda(tn_h/sig, tn_v/sig, dtn_h_lh, dtn_v_lh,lambda_v/sig);
            [d1dptn_h_lv, d1dptn_v_lv] = dprox_L12(tn_h/sig, tn_v/sig,dtn_h_lv, dtn_v_lv, lambda_v/sig);
            [d1dptn_h_lh, d1dptn_v_lh] = dprox_L12(tn_h/sig, tn_v/sig,dtn_h_lh, dtn_v_lh, lambda_v/sig);
            [ptn_h, ptn_v] = prox_L12(tn_h/sig, tn_v/sig, lambda_v/sig);
            [Edptn_h_lv, Edptn_v_lv] = dprox_L12_lambda(Etn_h/sig, Etn_v/sig, lambda_v/sig);
            %[Edptn_h_lh, Edptn_v_lh] = dprox_L12_lambda(Etn_h/sig, Etn_v/sig,Edtn_h_lh, Edtn_v_lh, lambda_v/sig);
            [d1Edptn_h_lv, d1Edptn_v_lv] = dprox_L12(Etn_h/sig, Etn_v/sig,Edtn_h_lv, Edtn_v_lv, lambda_v/sig);
            [d1Edptn_h_lh, d1Edptn_v_lh] = dprox_L12(Etn_h/sig, Etn_v/sig,Edtn_h_lh, Edtn_v_lh, lambda_v/sig);
            [Eptn_h, Eptn_v] = prox_L12(Etn_h/sig, Etn_v/sig, lambda_v/sig);
            
            dtn_h_lv=dtn_h_lv - dptn_h_lv - d1dptn_h_lv;
            dtn_h_lh=dtn_h_lh - d1dptn_h_lh;
            tn_h=tn_h - sig*ptn_h;
            Edtn_h_lv=Edtn_h_lv - Edptn_h_lv - d1Edptn_h_lv;
            Edtn_h_lh=Edtn_h_lh - d1Edptn_h_lh;
            Etn_h=Etn_h - sig*Eptn_h;
            
            dtn_v_lv=dtn_v_lv - dptn_v_lv - d1dptn_v_lv;
            dtn_v_lh=dtn_v_lh - d1dptn_v_lh;
            tn_v=tn_v - sig*ptn_v;
            Edtn_v_lv=Edtn_v_lv - Edptn_v_lv - d1Edptn_v_lv;
            Edtn_v_lh=Edtn_v_lh - d1Edptn_v_lh;
            Etn_v=Etn_v - sig*Eptn_v;
            
            %Update of the descent steps
            theta = (1+2*mu_g*tau)^(-1/2);
            tau = theta*tau;
            sig=sig/theta;
            
            %Update dual auxiliary variable
            dbgn_h_lv = dgn_h_lv + theta*(dgn_h_lv-dgn_hs_lv);
            dbgn_h_lh = dgn_h_lh + theta*(dgn_h_lh-dgn_hs_lh);
            bgn_h = gn_h + theta*(gn_h-gn_hs);
            Edbgn_h_lv = Edgn_h_lv + theta*(Edgn_h_lv-Edgn_hs_lv);
            Edbgn_h_lh = Edgn_h_lh + theta*(Edgn_h_lh-Edgn_hs_lh);
            Ebgn_h = Egn_h + theta*(Egn_h-Egn_hs);
            
            dbgn_v_lv = dgn_v_lv + theta*(dgn_v_lv-dgn_vs_lv);
            dbgn_v_lh = dgn_v_lh + theta*(dgn_v_lh-dgn_vs_lh);
            bgn_v = gn_v + theta*(gn_v-gn_vs);
            Edbgn_v_lv = Edgn_v_lv + theta*(Edgn_v_lv-Edgn_vs_lv);
            Edbgn_v_lh = Edgn_v_lh + theta*(Edgn_v_lh-Edgn_vs_lh);
            Ebgn_v = Egn_v + theta*(Egn_v-Egn_vs);
            
            dbtn_h_lv = dtn_h_lv + theta*(dtn_h_lv-dtn_hs_lv);
            dbtn_h_lh = dtn_h_lh + theta*(dtn_h_lh-dtn_hs_lh);
            btn_h = tn_h + theta*(tn_h-tn_hs);
            Edbtn_h_lv = Edtn_h_lv + theta*(Edtn_h_lv-Edtn_hs_lv);
            Edbtn_h_lh = Edtn_h_lh + theta*(Edtn_h_lh-Edtn_hs_lh);
            Ebtn_h = Etn_h + theta*(Etn_h-Etn_hs);
            
            dbtn_v_lv = dtn_v_lv + theta*(dtn_v_lv-dtn_vs_lv);
            dbtn_v_lh = dtn_v_lh + theta*(dtn_v_lh-dtn_vs_lh);
            btn_v = tn_v + theta*(tn_v-tn_vs);
            Edbtn_v_lv = Edtn_v_lv + theta*(Edtn_v_lv-Edtn_vs_lv);
            Edbtn_v_lh = Edtn_v_lh + theta*(Edtn_v_lh-Edtn_vs_lh);
            Ebtn_v = Etn_v + theta*(Etn_v-Etn_vs);
            
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
    
    
        % Local regularity
        init_PD.dh_lv = dh_lv;
        init_PD.dh_lh = dh_lh;
        init_PD.h = h;
        init_PD.Edh_lv = Edh_lv;
        init_PD.Edh_lh = Edh_lh;
        init_PD.Eh = Eh;
        
        init_PD.dgn_h_lv = dgn_h_lv;
        init_PD.dgn_v_lv = dgn_v_lv;
        init_PD.dgn_h_lh = dgn_h_lh;
        init_PD.dgn_v_lh = dgn_v_lh;
        init_PD.gn_h = gn_h;
        init_PD.gn_v = gn_v;
        init_PD.Edgn_h_lv = Edgn_h_lv;
        init_PD.Edgn_v_lv = Edgn_v_lv;
        init_PD.Edgn_h_lh = Edgn_h_lh;
        init_PD.Edgn_v_lh = Edgn_v_lh;
        init_PD.Egn_h = Egn_h;
        init_PD.Egn_v = Egn_v;
        
        
        init_PD.dbgn_h_lv = dbgn_h_lv;
        init_PD.dbgn_v_lv = dbgn_v_lv;
        init_PD.dbgn_h_lh = dbgn_h_lh;
        init_PD.dbgn_v_lh = dbgn_v_lh;
        init_PD.bgn_h = bgn_h;
        init_PD.bgn_v = bgn_v;
        init_PD.Edbgn_h_lv = Edbgn_h_lv;
        init_PD.Edbgn_v_lv = Edbgn_v_lv;
        init_PD.Edbgn_h_lh = Edbgn_h_lh;
        init_PD.Edbgn_v_lh = Edbgn_v_lh;
        init_PD.Ebgn_h = Ebgn_h;
        init_PD.Ebgn_v = Ebgn_v;
        
        
        
        % Local variance
        init_PD.dv_lv = dv_lv;
        init_PD.dv_lh = dv_lh;
        init_PD.v = v;
        init_PD.Edv_lv = Edv_lv;
        init_PD.Edv_lh = Edv_lh;
        init_PD.Ev = Ev;
        
        init_PD.dtn_h_lv = dtn_h_lv;
        init_PD.dtn_v_lv = dtn_v_lv;
        init_PD.dtn_h_lh = dtn_h_lh;
        init_PD.dtn_v_lh = dtn_v_lh;
        init_PD.tn_h = tn_h;
        init_PD.tn_v = tn_v;
        init_PD.Edtn_h_lv = Edtn_h_lv;
        init_PD.Edtn_v_lv = Edtn_v_lv;
        init_PD.Edtn_h_lh = Edtn_h_lh;
        init_PD.Edtn_v_lh = Edtn_v_lh;
        init_PD.Etn_h = Etn_h;
        init_PD.Etn_v = Etn_v;
        
        
        init_PD.dbtn_h_lv = dbtn_h_lv;
        init_PD.dbtn_v_lv = dbtn_v_lv;
        init_PD.dbtn_h_lh = dbtn_h_lh;
        init_PD.dbtn_v_lh = dbtn_v_lh;
        init_PD.btn_h = btn_h;
        init_PD.btn_v = btn_v;
        init_PD.Edbtn_h_lv = Edbtn_h_lv;
        init_PD.Edbtn_v_lv = Edbtn_v_lv;
        init_PD.Edbtn_h_lh = Edbtn_h_lh;
        init_PD.Edbtn_v_lh = Edbtn_v_lh;
        init_PD.Ebtn_h = Ebtn_h;
        init_PD.Ebtn_v = Ebtn_v;
        
        set_init(init_PD);
end