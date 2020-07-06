% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize the joint functional for fractal process segmentation with 
% iterative differentiation w.r.t. the regularization parameters (signal)
%
% from B. Pascal, N. Pustelnik, P. Abry: How Joint Fractal Features Estimation 
% and Texture Segmentation can be cast into a Strongly Convex Optimization
% Problem ?. arxiv:1910.05246
% and A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging. J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)
% and B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [v, h, Ev, Eh, dv_lv, dv_lh, dh_lv, dh_lh, Edv_lv, Edv_lh, Edh_lv, Edh_lh, crit, gap, t]=dPA_PDj_1D(L_X,prox,sure)
    
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
    normD = 1;
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
        
        dgn_lv = init_PD.dgn_lv;
        dgn_lh = init_PD.dgn_lh;
        gn = init_PD.gn;
        Edgn_lv = init_PD.Edgn_lv;
        Edgn_lh = init_PD.Edgn_lh;
        Egn = init_PD.Egn;
        
        
        dbgn_lv = init_PD.dbgn_lv;
        dbgn_lh = init_PD.dbgn_lh;
        bgn = init_PD.bgn;
        Edbgn_lv = init_PD.Edbgn_lv;
        Edbgn_lh = init_PD.Edbgn_lh;
        Ebgn = init_PD.Ebgn;
        
        
        
        % Local variance
        dv_lv = init_PD.dv_lv;
        dv_lh = init_PD.dv_lh;
        v = init_PD.v;
        Edv_lv = init_PD.Edv_lv;
        Edv_lh = init_PD.Edv_lh;
        Ev = init_PD.Ev;
        
        dtn_lv = init_PD.dtn_lv;
        dtn_lh = init_PD.dtn_lh;
        tn = init_PD.tn;
        Edtn_lv = init_PD.Edtn_lv;
        Edtn_lh = init_PD.Edtn_lh;
        Etn = init_PD.Etn;
        
        
        dbtn_lv = init_PD.dbtn_lv;
        dbtn_lh = init_PD.dbtn_lh;
        btn = init_PD.btn;
        Edbtn_lv = init_PD.Edbtn_lv;
        Edbtn_lh = init_PD.Edbtn_lh;
        Ebtn = init_PD.Ebtn;
        
    else
        
    % Local regularity
    dh_lv = zeros(size(L{JJ(1)}));
    dh_lh = zeros(size(L{JJ(1)}));
    h = zeros(size(L{JJ(1)}));
    Edh_lv = zeros(size(L{JJ(1)}));
    Edh_lh = zeros(size(L{JJ(1)}));
    Eh = zeros(size(L{JJ(1)}));
    
    dgn_lv = opL_1D(dh_lv);
    dgn_lh = opL_1D(dh_lh);
    gn = opL_1D(h);
    Edgn_lv = opL_1D(Edh_lv);
    Edgn_lh = opL_1D(Edh_lh);
    Egn = opL_1D(Eh);
    
    dbgn_lv = dgn_lv;
    dbgn_lh = dgn_lh;
    bgn = gn;
    Edbgn_lv = Edgn_lv;
    Edbgn_lh = Edgn_lh;
    Ebgn = Egn;
    
    
    % Local variance
    dv_lv = zeros(size(L{JJ(1)}));
    dv_lh = zeros(size(L{JJ(1)}));
    v = zeros(size(L{JJ(1)}));
    Edv_lv = zeros(size(L{JJ(1)}));
    Edv_lh = zeros(size(L{JJ(1)}));
    Ev = zeros(size(L{JJ(1)}));
    
    dtn_lv = opL_1D(dv_lv);
    dtn_lh = opL_1D(dv_lh);
    tn = opL_1D(v);
    Edtn_lv = opL_1D(Edv_lv);
    Edtn_lh = opL_1D(Edv_lh);
    Etn = opL_1D(Ev);
    
    dbtn_lv = dtn_lv;
    dbtn_lh = dtn_lh;
    btn = tn;
    Edbtn_lv = Edtn_lv;
    Edbtn_lh = Edtn_lh;
    Ebtn = Etn;
    
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
            dgns_lv = dgn_lv;
            dgns_lh = dgn_lh;
            gns = gn;
            Edgns_lv = Edgn_lv;
            Edgns_lh = Edgn_lh;
            Egns = Egn;
            
            dtns_lv = dtn_lv;
            dtns_lh = dtn_lh;
            tns = tn;
            Edtns_lv = Edtn_lv;
            Edtns_lh = Edtn_lh;
            Etns = Etn;
            
            
            %Update of primal variable
            dh_lv = dh_lv - tau*opL_adj_1D(dbgn_lv);
            dh_lh = dh_lh - tau*opL_adj_1D(dbgn_lh);
            h = h - tau*opL_adj_1D(bgn);
            Edh_lv = Edh_lv - tau*opL_adj_1D(Edbgn_lv);
            Edh_lh = Edh_lh - tau*opL_adj_1D(Edbgn_lh);
            Eh = Eh - tau*opL_adj_1D(Ebgn);
            
            dv_lv = dv_lv - tau*opL_adj_1D(dbtn_lv);
            dv_lh = dv_lh - tau*opL_adj_1D(dbtn_lh);
            v = v - tau*opL_adj_1D(btn);
            Edv_lv = Edv_lv - tau*opL_adj_1D(Edbtn_lv);
            Edv_lh = Edv_lh - tau*opL_adj_1D(Edbtn_lh);
            Ev = Ev - tau*opL_adj_1D(Ebtn);
            
            [dh_lv, dv_lv] = prox_hv(dh_lv,dv_lv,use_0,tau);
            [dh_lh, dv_lh] = prox_hv(dh_lh,dv_lh,use_0,tau);
            [h, v] = prox_hv(h,v,use,tau);
            [Edh_lv, Edv_lv] = prox_hv(Edh_lv,Edv_lv,use_0,tau);
            [Edh_lh, Edv_lh] = prox_hv(Edh_lh,Edv_lh,use_0,tau);
            [Eh, Ev] = prox_hv(Eh,Ev,use_per,tau);
            
            %Update of dual variables
            
            %Dual Variable of h
            dpgn_lv = opL_1D(dh_lv);
            dpgn_lh = opL_1D(dh_lh);
            pgn = opL_1D(h);
            Edpgn_lv = opL_1D(Edh_lv);
            Edpgn_lh = opL_1D(Edh_lh);
            Epgn = opL_1D(Eh);
            
            dgn_lv = dgn_lv + sig*dpgn_lv; %prox argument
            dgn_lh = dgn_lh + sig*dpgn_lh; %prox argument
            gn = gn + sig*pgn; %prox argument
            Edgn_lv = Edgn_lv + sig*Edpgn_lv; %prox argument
            Edgn_lh = Edgn_lh + sig*Edpgn_lh; %prox argument
            Egn = Egn + sig*Epgn; %prox argument
            
            
            d1dpgn_lv = dprox_L1(gn/sig, dgn_lv, lambda_h/sig);
            d1dpgn_lh = dprox_L1(gn/sig, dgn_lh, lambda_h/sig);
            dpgn_lh = dprox_L1_lambda(gn/sig, lambda_h/sig);
            pgn = prox_L1(gn/sig, lambda_h/sig);
            d1Edpgn_lv = dprox_L1(Egn/sig, Edgn_lv, lambda_h/sig);
            d1Edpgn_lh = dprox_L1(Egn/sig, Edgn_lh, lambda_h/sig);
            Edpgn_lh = dprox_L1_lambda(Egn/sig, lambda_h/sig);
            Epgn = prox_L1(Egn/sig, lambda_h/sig);
            
            dgn_lv=dgn_lv - d1dpgn_lv;
            dgn_lh=dgn_lh - dpgn_lh - d1dpgn_lh;
            gn=gn - sig*pgn;
            Edgn_lv=Edgn_lv - d1Edpgn_lv;
            Edgn_lh=Edgn_lh - Edpgn_lh - d1Edpgn_lh;
            Egn=Egn - sig*Epgn;
            
            
            %Dual Variable of v
            dptn_lv = opL_1D(dv_lv);
            dptn_lh = opL_1D(dv_lh);
            ptn = opL_1D(v);
            Edptn_lv = opL_1D(Edv_lv);
            Edptn_lh = opL_1D(Edv_lh);
            Eptn = opL_1D(Ev);
            
            dtn_lv = dtn_lv + sig*dptn_lv; %prox argument
            dtn_lh = dtn_lh + sig*dptn_lh; %prox argument
            tn = tn + sig*ptn; %prox argument
            Edtn_lv = Edtn_lv + sig*Edptn_lv; %prox argument
            Edtn_lh = Edtn_lh + sig*Edptn_lh; %prox argument
            Etn = Etn + sig*Eptn; %prox argument
            
            
            d1dptn_lv = dprox_L1(tn/sig, dtn_lv, lambda_v/sig);
            d1dptn_lh = dprox_L1(tn/sig, dtn_lh, lambda_v/sig);
            dptn_lv = dprox_L1_lambda(tn/sig, lambda_v/sig);
            ptn = prox_L1(tn/sig, lambda_v/sig);
            d1Edptn_lv = dprox_L1(Etn/sig, Edtn_lv, lambda_v/sig);
            d1Edptn_lh = dprox_L1(Etn/sig, Edtn_lh, lambda_v/sig);
            Edptn_lv = dprox_L1_lambda(Etn/sig, lambda_v/sig);
            Eptn = prox_L1(Etn/sig, lambda_v/sig);
            
            dtn_lv=dtn_lv - dptn_lv - d1dptn_lv;
            dtn_lh=dtn_lh - d1dptn_lh;
            tn=tn - sig*ptn;
            Edtn_lv=Edtn_lv - Edptn_lv - d1Edptn_lv;
            Edtn_lh=Edtn_lh - d1Edptn_lh;
            Etn=Etn - sig*Eptn;
            
           
            %Update of the descent steps
            theta = (1+2*mu_g*tau)^(-1/2);
            tau = theta*tau;
            sig=sig/theta;
            
            %Update dual auxiliary variable
            dbgn_lv = dgn_lv + theta*(dgn_lv-dgns_lv);
            dbgn_lh = dgn_lh + theta*(dgn_lh-dgns_lh);
            bgn = gn + theta*(gn-gns);
            Edbgn_lv = Edgn_lv + theta*(Edgn_lv-Edgns_lv);
            Edbgn_lh = Edgn_lh + theta*(Edgn_lh-Edgns_lh);
            Ebgn = Egn + theta*(Egn-Egns);
            
            
            dbtn_lv = dtn_lv + theta*(dtn_lv-dtns_lv);
            dbtn_lh = dtn_lh + theta*(dtn_lh-dtns_lh);
            btn = tn + theta*(tn-tns);
            Edbtn_lv = Edtn_lv + theta*(Edtn_lv-Edtns_lv);
            Edbtn_lh = Edtn_lh + theta*(Edtn_lh-Edtns_lh);
            Ebtn = Etn + theta*(Etn-Etns);
            
            
            t(it) = toc;
            
            %% Compute convergence criteria
            
            [c,~,g] = PDj_gap_1D(v, h, tn, gn, L, adj_tv, adj_th, f_adj0, JJ, prox, use);
            
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
        
        init_PD.dgn_lv = dgn_lv;
        init_PD.dgn_lh = dgn_lh;
        init_PD.gn = gn;
        init_PD.Edgn_lv = Edgn_lv;
        init_PD.Edgn_lh = Edgn_lh;
        init_PD.Egn = Egn;
        
        
        init_PD.dbgn_lv = dbgn_lv;
        init_PD.dbgn_lh = dbgn_lh;
        init_PD.bgn = bgn;
        init_PD.Edbgn_lv = Edbgn_lv;
        init_PD.Edbgn_lh = Edbgn_lh;
        init_PD.Ebgn = Ebgn;
        
        
        
        % Local variance
        init_PD.dv_lv = dv_lv;
        init_PD.dv_lh = dv_lh;
        init_PD.v = v;
        init_PD.Edv_lv = Edv_lv;
        init_PD.Edv_lh = Edv_lh;
        init_PD.Ev = Ev;
        
        init_PD.dtn_lv = dtn_lv;
        init_PD.dtn_lh = dtn_lh;
        init_PD.tn = tn;
        init_PD.Edtn_lv = Edtn_lv;
        init_PD.Edtn_lh = Edtn_lh;
        init_PD.Etn = Etn;
        
        
        init_PD.dbtn_lv = dbtn_lv;
        init_PD.dbtn_lh = dbtn_lh;
        init_PD.btn = btn;
        init_PD.Edbtn_lv = Edbtn_lv;
        init_PD.Edbtn_lh = Edbtn_lh;
        init_PD.Ebtn = Ebtn;
        
        set_init(init_PD);
    
end