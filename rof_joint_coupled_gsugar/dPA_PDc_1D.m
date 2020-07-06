% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize the coupled functional for for fractal process segmentation with 
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

function [v, h, Ev, Eh, dv_lv, dv_la, dh_lv, dh_la, Edv_lv, Edv_la, Edh_lv, Edh_la, crit, gap, t]=dPA_PDc_1D(L_X,prox,sure)
    
    % inputs  - L_X.leaders: undecimated wavelet leaders
    %         - prox: (lambda, alpha) regularization parameters
    %         - sure: delta (Monte Carlo vector) eps (Finite Difference step)
    %
    % ouputs  - v: regularized estimate of local power
    %         - h: regularized estimate of local regularity
    %         - Ev: perturbed regularized estimate of local power
    %         - Eh: perturbed regularized estimate of local regularity
    %         - dv_lv: gradient of regularized estimate of local power
    %         w.r.t. regularization parameter lambda
    %         - dv_la: gradient of regularized estimate of local power
    %         w.r.t. regularization parameter alpha
    %         - dh_lv: gradient of regularized estimate of local regularity
    %         w.r.t. regularization parameter lambda
    %         - dh_la: gradient of regularized estimate of local regularity
    %         w.r.t. regularization parameter alpha
    %         - Edv_lv: gradient of perturbed regularized estimate of local power
    %         w.r.t. regularization parameter lambda
    %         - Edv_la: gradient of perturbed regularized estimate of local power
    %         w.r.t. regularization parameter alpha
    %         - Edh_lv: gradient of perturbed regularized estimate of local regularity
    %         w.r.t. regularization parameter lambda
    %         - Edh_la: gradient of perturbed regularized estimate of local regularity
    %         w.r.t. regularization parameter alpha
    %         - t: time
    %         - crit: evolution of objective function
    %         - gap: evolution of duality gap
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    %% From leaders to log-leaders
    leaders = L_X.leaders;
    J1=L_X.JJ(1);
    J2=L_X.JJ(end);
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
    iter = 1e5;
    eps = 1e-4;
    
    %% Initialization
    
    init_PD = get_init;
    
    if isfield(init_PD,'h')
        
        % Local regularity
        dh_lv = init_PD.dh_lv;
        dh_la = init_PD.dh_la;
        h = init_PD.h;
        Edh_lv = init_PD.Edh_lv;
        Edh_la = init_PD.Edh_la;
        Eh = init_PD.Eh;
        
        dgn_lv = init_PD.dgn_lv;
        dgn_la = init_PD.dgn_la;
        gn = init_PD.gn;
        Edgn_lv = init_PD.Edgn_lv;
        Edgn_la = init_PD.Edgn_la;
        Egn = init_PD.Egn;
        
        
        dbgn_lv = init_PD.dbgn_lv;
        dbgn_la = init_PD.dbgn_la;
        bgn = init_PD.bgn;
        Edbgn_lv = init_PD.Edbgn_lv;
        Edbgn_la = init_PD.Edbgn_la;
        Ebgn = init_PD.Ebgn;
        
        
        
        % Local variance
        dv_lv = init_PD.dv_lv;
        dv_la = init_PD.dv_la;
        v = init_PD.v;
        Edv_lv = init_PD.Edv_lv;
        Edv_la = init_PD.Edv_la;
        Ev = init_PD.Ev;
        
        dtn_lv = init_PD.dtn_lv;
        dtn_la = init_PD.dtn_la;
        tn = init_PD.tn;
        Edtn_lv = init_PD.Edtn_lv;
        Edtn_la = init_PD.Edtn_la;
        Etn = init_PD.Etn;
        
        
        dbtn_lv = init_PD.dbtn_lv;
        dbtn_la = init_PD.dbtn_la;
        btn = init_PD.btn;
        Edbtn_lv = init_PD.Edbtn_lv;
        Edbtn_la = init_PD.Edbtn_la;
        Ebtn = init_PD.Ebtn;
        
    else
        
    dh_lv = zeros(size(L{J1}));
    dh_la = zeros(size(L{J1}));
    h = zeros(size(L{J1}));
    Edh_lv = zeros(size(L{J1}));
    Edh_la = zeros(size(L{J1}));
    Eh = zeros(size(L{J1}));
    
    dgn_lv = alpha*opL_1D(dh_lv);
    dgn_la = alpha*opL_1D(dh_la) + opL_1D(h);
    gn = alpha*opL_1D(h);
    Edgn_lv = alpha*opL_1D(Edh_lv);
    Edgn_la = alpha*opL_1D(Edh_la) + opL_1D(Eh);
    Egn = alpha*opL_1D(Eh);
    
    dbgn_lv = dgn_lv;
    dbgn_la = dgn_la;
    bgn = gn;
    Edbgn_lv = Edgn_lv;
    Edbgn_la = Edgn_la;
    Ebgn = Egn;
    
    dv_lv = zeros(size(L{J1}));
    dv_la = zeros(size(L{J1}));
    v = zeros(size(L{J1}));
    Edv_lv = zeros(size(L{J1}));
    Edv_la = zeros(size(L{J1}));
    Ev = zeros(size(L{J1}));
    
    dtn_lv = opL_1D(dv_lv);
    dtn_la = opL_1D(dv_la);
    tn = opL_1D(v);
    Edtn_lv = opL_1D(Edv_lv);
    Edtn_la = opL_1D(Edv_la);
    Etn = opL_1D(Ev);
    
    dbtn_lv = dtn_lv;
    dbtn_la = dtn_la;
    btn = tn;
    Edbtn_lv = Edtn_lv;
    Edbtn_la = Edtn_la;
    Ebtn = Etn;
    end 
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
            dgn_s_lv = dgn_lv;
            dgn_s_la = dgn_la;
            gn_s = gn;
            Edgn_s_lv = Edgn_lv;
            Edgn_s_la = Edgn_la;
            Egn_s = Egn;
            
            dtns_lv = dtn_lv;
            dtns_la = dtn_la;
            tns = tn;
            Edtns_lv = Edtn_lv;
            Edtns_la = Edtn_la;
            Etns = Etn;
            
            %Update of primal variable
            dh_lv = dh_lv - tau*alpha*opL_adj_1D(dbgn_lv);
            dh_la = dh_la - tau*alpha*opL_adj_1D(dbgn_la)- tau*opL_adj_1D(bgn);
            h = h - tau*alpha*opL_adj_1D(bgn);
            Edh_lv = Edh_lv - tau*alpha*opL_adj_1D(Edbgn_lv);
            Edh_la = Edh_la - tau*alpha*opL_adj_1D(Edbgn_la)- tau*opL_adj_1D(Ebgn);
            Eh = Eh - tau*alpha*opL_adj_1D(Ebgn);
            
            dv_lv = dv_lv - tau*opL_adj_1D(dbtn_lv);
            dv_la = dv_la - tau*opL_adj_1D(dbtn_la);
            v = v - tau*opL_adj_1D(btn);
            Edv_lv = Edv_lv - tau*opL_adj_1D(Edbtn_lv);
            Edv_la = Edv_la - tau*opL_adj_1D(Edbtn_la);
            Ev = Ev - tau*opL_adj_1D(Ebtn);
            
            [dh_lv, dv_lv] = prox_hv(dh_lv,dv_lv,use_0,tau);
            [dh_la, dv_la] = prox_hv(dh_la,dv_la,use_0,tau);
            [h, v] = prox_hv(h,v,use,tau);
            [Edh_lv, Edv_lv] = prox_hv(Edh_lv,Edv_lv,use_0,tau);
            [Edh_la, Edv_la] = prox_hv(Edh_la,Edv_la,use_0,tau);
            [Eh, Ev] = prox_hv(Eh,Ev,use_per,tau);
            
            %Update of dual variables
            
            dpgn_lv = alpha*opL_1D(dh_lv);
            dpgn_la = alpha*opL_1D(dh_la) + opL_1D(h);
            pgn = alpha*opL_1D(h);
            Edpgn_lv = alpha*opL_1D(Edh_lv);
            Edpgn_la = alpha*opL_1D(Edh_la) + opL_1D(Eh);
            Epgn = alpha*opL_1D(Eh);
            
            dptn_lv = opL_1D(dv_lv);
            dptn_la = opL_1D(dv_la);
            ptn = opL_1D(v);
            Edptn_lv = opL_1D(Edv_lv);
            Edptn_la = opL_1D(Edv_la);
            Eptn = opL_1D(Ev);
            
            dgn_lv = dgn_lv + sig*dpgn_lv;
            dgn_la = dgn_la + sig*dpgn_la;
            gn = gn + sig*pgn;
            Edgn_lv = Edgn_lv + sig*Edpgn_lv;
            Edgn_la = Edgn_la + sig*Edpgn_la;
            Egn = Egn + sig*Epgn;
            
            dtn_lv = dtn_lv + sig*dptn_lv;
            dtn_la = dtn_la + sig*dptn_la;
            tn = tn + sig*ptn;
            Edtn_lv = Edtn_lv + sig*Edptn_lv;
            Edtn_la = Edtn_la + sig*Edptn_la;
            Etn = Etn + sig*Eptn;
            
            [dpgn_lv, dptn_lv] = dprox_L1c(gn/sig, tn/sig,dgn_lv/sig, dtn_lv/sig, lambda/sig);
            [d1pgn_lv, d1ptn_lv] = dprox_L1c_lambda(gn/sig, tn/sig, lambda/sig);
            [dpgn_la, dptn_la] = dprox_L1c(gn/sig, tn/sig,dgn_la/sig, dtn_la/sig, lambda/sig);
            [pgn, ptn] = prox_L1c(gn/sig, tn/sig, lambda/sig);
            [Edpgn_lv, Edptn_lv] = dprox_L1c(Egn/sig, Etn/sig,Edgn_lv/sig, Edtn_lv/sig, lambda/sig);
            [Ed1pgn_lv, Ed1ptn_lv] = dprox_L1c_lambda(Egn/sig, Etn/sig, lambda/sig);
            [Edpgn_la, Edptn_la] = dprox_L1c(Egn/sig, Etn/sig,Edgn_la/sig, Edtn_la/sig, lambda/sig);
            [Epgn, Eptn] = prox_L1c(Egn/sig, Etn/sig, lambda/sig);
            
            dgn_lv = dgn_lv - sig*dpgn_lv - sig*d1pgn_lv;
            dgn_la = dgn_la - sig*dpgn_la;
            gn = gn - sig*pgn;
            Edgn_lv = Edgn_lv - sig*Edpgn_lv - sig*Ed1pgn_lv;
            Edgn_la = Edgn_la - sig*Edpgn_la;
            Egn = Egn - sig*Epgn;
            
            dtn_lv = dtn_lv - sig*dptn_lv - sig*d1ptn_lv;
            dtn_la = dtn_la - sig*dptn_la;
            tn = tn - sig*ptn;
            Edtn_lv = Edtn_lv - sig*Edptn_lv - sig*Ed1ptn_lv;
            Edtn_la = Edtn_la - sig*Edptn_la;
            Etn = Etn - sig*Eptn;
            
            
            %Update of the descent steps
            theta = (1+2*mu_g*tau)^(-1/2);
            tau = theta*tau;
            sig=sig/theta;
            
            %Update dual auxiliary variable
            dbgn_lv = dgn_lv + theta*(dgn_lv-dgn_s_lv);
            dbgn_la = dgn_la + theta*(dgn_la-dgn_s_la);
            bgn = gn + theta*(gn-gn_s);
            Edbgn_lv = Edgn_lv + theta*(Edgn_lv-Edgn_s_lv);
            Edbgn_la = Edgn_la + theta*(Edgn_la-Edgn_s_la);
            Ebgn = Egn + theta*(Egn-Egn_s);
            
            dbtn_lv = dtn_lv + theta*(dtn_lv-dtns_lv);
            dbtn_la = dtn_la + theta*(dtn_la-dtns_la);
            btn = tn + theta*(tn-tns);
            Edbtn_lv = Edtn_lv + theta*(Edtn_lv-Edtns_lv);
            Edbtn_la = Edtn_la + theta*(Edtn_la-Edtns_la);
            Ebtn = Etn + theta*(Etn-Etns);
            
            
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
    
    % Local regularity
        init_PD.dh_lv = dh_lv;
        init_PD.dh_la = dh_la;
        init_PD.h = h;
        init_PD.Edh_lv = Edh_lv;
        init_PD.Edh_la = Edh_la;
        init_PD.Eh = Eh;
        
        init_PD.dgn_lv = dgn_lv;
        init_PD.dgn_la = dgn_la;
        init_PD.gn = gn;
        init_PD.Edgn_lv = Edgn_lv;
        init_PD.Edgn_la = Edgn_la;
        init_PD.Egn = Egn;
        
        
        init_PD.dbgn_lv = dbgn_lv;
        init_PD.dbgn_la = dbgn_la;
        init_PD.bgn = bgn;
        init_PD.Edbgn_lv = Edbgn_lv;
        init_PD.Edbgn_la = Edbgn_la;
        init_PD.Ebgn = Ebgn;
        
        
        
        % Local variance
        init_PD.dv_lv = dv_lv;
        init_PD.dv_la = dv_la;
        init_PD.v = v;
        init_PD.Edv_lv = Edv_lv;
        init_PD.Edv_la = Edv_la;
        init_PD.Ev = Ev;
        
        init_PD.dtn_lv = dtn_lv;
        init_PD.dtn_la = dtn_la;
        init_PD.tn = tn;
        init_PD.Edtn_lv = Edtn_lv;
        init_PD.Edtn_la = Edtn_la;
        init_PD.Etn = Etn;
        
        
        init_PD.dbtn_lv = dbtn_lv;
        init_PD.dbtn_la = dbtn_la;
        init_PD.btn = btn;
        init_PD.Edbtn_lv = Edbtn_lv;
        init_PD.Edbtn_la = Edbtn_la;
        init_PD.Ebtn = Ebtn;
        
        set_init(init_PD);
        
end