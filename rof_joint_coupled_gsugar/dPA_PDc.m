% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to minimize the coupled functional for fractal texture segmentation with 
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



function [v, h, Ev, Eh, dv_lv, dv_la, dh_lv, dh_la, Edv_lv, Edv_la, Edh_lv, Edh_la, crit, gap, t]=dPA_PDc(L_X,prox,sure)
    
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
    JJ = L_X.JJ;
    L_0 = cell(1,JJ(end));
    L = cell(1,JJ(end));
    L_per = cell(1,JJ(end));
    for jj=JJ
        L_0{jj} = zeros(size(leaders{jj}));
        L{jj} = log2(leaders{jj});
        L_per{jj} = log2(leaders{jj}) + sure.eps*sure.delta{jj};
    end
    
    %% Usefull quantities for prox data fid
    use_0 = usefull(JJ, L_0);
    use = usefull(JJ, L);
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
    mu_g=use.mu;
    alpha = prox.alpha;
    use.alpha = alpha;
    normD = max(1,alpha)*sqrt(2);
    Lip = normD;
    tau = gamma/Lip;
    sig = gamma/Lip;
    lambda = prox.lambda;
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
        
        dg_h_lv = init_PD.dg_h_lv;
        dg_v_lv = init_PD.dg_v_lv;
        dg_h_la = init_PD.dg_h_la;
        dg_v_la = init_PD.dg_v_la;
        g_h = init_PD.g_h;
        g_v = init_PD.g_v;
        Edg_h_lv = init_PD.Edg_h_lv;
        Edg_v_lv = init_PD.Edg_v_lv;
        Edg_h_la = init_PD.Edg_h_la;
        Edg_v_la = init_PD.Edg_v_la;
        Eg_h = init_PD.Eg_h;
        Eg_v = init_PD.Eg_v;
        
        
        dbg_h_lv = init_PD.dbg_h_lv;
        dbg_v_lv = init_PD.dbg_v_lv;
        dbg_h_la = init_PD.dbg_h_la;
        dbg_v_la = init_PD.dbg_v_la;
        bg_h = init_PD.bg_h;
        bg_v = init_PD.bg_v;
        Edbg_h_lv = init_PD.Edbg_h_lv;
        Edbg_v_lv = init_PD.Edbg_v_lv;
        Edbg_h_la = init_PD.Edbg_h_la;
        Edbg_v_la = init_PD.Edbg_v_la;
        Ebg_h = init_PD.Ebg_h;
        Ebg_v = init_PD.Ebg_v;
        
        
        
        % Local variance
        dv_lv = init_PD.dv_lv;
        dv_la = init_PD.dv_la;
        v = init_PD.v;
        Edv_lv = init_PD.Edv_lv;
        Edv_la = init_PD.Edv_la;
        Ev = init_PD.Ev;
        
        dt_h_lv = init_PD.dt_h_lv;
        dt_v_lv = init_PD.dt_v_lv;
        dt_h_la = init_PD.dt_h_la;
        dt_v_la = init_PD.dt_v_la;
        t_h = init_PD.t_h;
        t_v = init_PD.t_v;
        Edt_h_lv = init_PD.Edt_h_lv;
        Edt_v_lv = init_PD.Edt_v_lv;
        Edt_h_la = init_PD.Edt_h_la;
        Edt_v_la = init_PD.Edt_v_la;
        Et_h = init_PD.Et_h;
        Et_v = init_PD.Et_v;
        
        
        dbt_h_lv = init_PD.dbt_h_lv;
        dbt_v_lv = init_PD.dbt_v_lv;
        dbt_h_la = init_PD.dbt_h_la;
        dbt_v_la = init_PD.dbt_v_la;
        bt_h = init_PD.bt_h;
        bt_v = init_PD.bt_v;
        Edbt_h_lv = init_PD.Edbt_h_lv;
        Edbt_v_lv = init_PD.Edbt_v_lv;
        Edbt_h_la = init_PD.Edbt_h_la;
        Edbt_v_la = init_PD.Edbt_v_la;
        Ebt_h = init_PD.Ebt_h;
        Ebt_v = init_PD.Ebt_v;
        
    else
        
        % Local regularity
        dh_lv = zeros(size(L{JJ(1)}));
        dh_la = zeros(size(L{JJ(1)}));
        h = zeros(size(L{JJ(1)}));
        Edh_lv = zeros(size(L{JJ(1)}));
        Edh_la = zeros(size(L{JJ(1)}));
        Eh = zeros(size(L{JJ(1)}));
        
        [dg_h_lv, dg_v_lv] = opL(dh_lv);
        [dg_h_la, dg_v_la] = opL(dh_la);
        [g_h, g_v] = opL(h);
        [Edg_h_lv, Edg_v_lv] = opL(Edh_lv);
        [Edg_h_la, Edg_v_la] = opL(Edh_la);
        [Eg_h, Eg_v] = opL(Eh);
        
        dbg_h_lv = dg_h_lv;
        dbg_h_la = dg_h_la;
        bg_h = g_h;
        Edbg_h_lv = Edg_h_lv;
        Edbg_h_la = Edg_h_la;
        Ebg_h = Eg_h;
        
        dbg_v_lv = dg_v_lv;
        dbg_v_la = dg_v_la;
        bg_v = g_v;
        Edbg_v_lv = Edg_v_lv;
        Edbg_v_la = Edg_v_la;
        Ebg_v = Eg_v;
        
        dv_lv = zeros(size(L{JJ(1)}));
        dv_la = zeros(size(L{JJ(1)}));
        v = zeros(size(L{JJ(1)}));
        Edv_lv = zeros(size(L{JJ(1)}));
        Edv_la = zeros(size(L{JJ(1)}));
        Ev = zeros(size(L{JJ(1)}));
        
        [dt_h_lv, dt_v_lv] = opL(dv_lv);
        [dt_h_la, dt_v_la] = opL(dv_la);
        [t_h, t_v] = opL(v);
        [Edt_h_lv, Edt_v_lv] = opL(Edv_lv);
        [Edt_h_la, Edt_v_la] = opL(Edv_la);
        [Et_h, Et_v] = opL(Ev);
        
        dbt_h_lv = dt_h_lv;
        dbt_h_la = dt_h_la;
        bt_h = t_h;
        Edbt_h_lv = Edt_h_lv;
        Edbt_h_la = Edt_h_la;
        Ebt_h = Et_h;
        
        dbt_v_lv = dt_v_lv;
        dbt_v_la = dt_v_la;
        bt_v = t_v;
        Edbt_v_lv = Edt_v_lv;
        Edbt_v_la = Edt_v_la;
        Ebt_v = Et_v;
        
    end
    
    crit = zeros(1,iter); gap = crit;
    t = crit;
    
    it = 0;
    gapc = eps + 1;
    
    %% ALGORITHM
    
    if lambda < 0 || alpha < 0
        [v,h] = linear_reg(L,L_X.JJ);
        [Ev,Eh] = linear_reg(L_per,L_X.JJ);
    else
        while (gapc > eps)&&(it<iter)
            
            it = it + 1;
            
            tic
            %Save the dual variables
            dg_hs_lv = dg_h_lv;
            dg_hs_la = dg_h_la;
            g_hs = g_h;
            Edg_hs_lv = Edg_h_lv;
            Edg_hs_la = Edg_h_la;
            Eg_hs = Eg_h;
            
            dg_vs_lv = dg_v_lv;
            dg_vs_la = dg_v_la;
            g_vs = g_v;
            Edg_vs_lv = Edg_v_lv;
            Edg_vs_la = Edg_v_la;
            Eg_vs = Eg_v;
            
            dt_hs_lv = dt_h_lv;
            dt_hs_la = dt_h_la;
            t_hs = t_h;
            Edt_hs_lv = Edt_h_lv;
            Edt_hs_la = Edt_h_la;
            Et_hs = Et_h;
            
            dt_vs_lv = dt_v_lv;
            dt_vs_la = dt_v_la;
            t_vs = t_v;
            Edt_vs_lv = Edt_v_lv;
            Edt_vs_la = Edt_v_la;
            Et_vs = Et_v;
            
            %Update of primal variable
            dh_lv = dh_lv - tau*alpha*opL_adj(dbg_h_lv,dbg_v_lv);
            dh_la = dh_la - tau*alpha*opL_adj(dbg_h_la,dbg_v_la) - tau*opL_adj(bg_h,bg_v);
            h = h - tau*alpha*opL_adj(bg_h,bg_v);
            Edh_lv = Edh_lv - tau*alpha*opL_adj(Edbg_h_lv,Edbg_v_lv);
            Edh_la = Edh_la - tau*alpha*opL_adj(Edbg_h_la,Edbg_v_la) - tau*opL_adj(Ebg_h,Ebg_v);
            Eh = Eh - tau*alpha*opL_adj(Ebg_h,Ebg_v);
            
            
            dv_lv = dv_lv - tau*opL_adj(dbt_h_lv,dbt_v_lv);
            dv_la = dv_la - tau*opL_adj(dbt_h_la,dbt_v_la);
            v = v - tau*opL_adj(bt_h,bt_v);
            Edv_lv = Edv_lv - tau*opL_adj(Edbt_h_lv,Edbt_v_lv);
            Edv_la = Edv_la - tau*opL_adj(Edbt_h_la,Edbt_v_la);
            Ev = Ev - tau*opL_adj(Ebt_h,Ebt_v);
            
            [dh_lv, dv_lv] = prox_hv(dh_lv,dv_lv,use_0,tau);
            [dh_la, dv_la] = prox_hv(dh_la,dv_la,use_0,tau);
            [h, v] = prox_hv(h,v,use,tau);
            [Edh_lv, Edv_lv] = prox_hv(Edh_lv,Edv_lv,use_0,tau);
            [Edh_la, Edv_la] = prox_hv(Edh_la,Edv_la,use_0,tau);
            [Eh, Ev] = prox_hv(Eh,Ev,use_per,tau);
            
            %Update of dual variables
            
            %Dual Variable of h
            [dpgn_h_lv, dpgn_v_lv] = opL(dh_lv);
            [dpgn_h_la, dpgn_v_la] = opL(dh_la);
            [pgn_h, pgn_v] = opL(h);
            [Edpgn_h_lv, Edpgn_v_lv] = opL(Edh_lv);
            [Edpgn_h_la, Edpgn_v_la] = opL(Edh_la);
            [Epgn_h, Epgn_v] = opL(Eh);
            
            dg_h_lv = dg_h_lv + sig*alpha*dpgn_h_lv; %prox argument
            dg_h_la = dg_h_la + sig*alpha*dpgn_h_la + sig*pgn_h; %prox argument
            g_h = g_h + sig*alpha*pgn_h; %prox argument
            Edg_h_lv = Edg_h_lv + sig*alpha*Edpgn_h_lv; %prox argument
            Edg_h_la = Edg_h_la + sig*alpha*Edpgn_h_la + sig*Epgn_h; %prox argument
            Eg_h = Eg_h + sig*alpha*Epgn_h; %prox argument
            
            dg_v_lv = dg_v_lv + sig*alpha*dpgn_v_lv; %prox argument
            dg_v_la = dg_v_la + sig*alpha*dpgn_v_la + sig*pgn_v; %prox argument
            g_v = g_v + sig*alpha*pgn_v; %prox argument
            Edg_v_lv = Edg_v_lv + sig*alpha*Edpgn_v_lv; %prox argument
            Edg_v_la = Edg_v_la + sig*alpha*Edpgn_v_la + sig*Epgn_v; %prox argument
            Eg_v = Eg_v + sig*alpha*Epgn_v; %prox argument
            
            
            %Dual Variable of s
            [dptn_h_lv, dptn_v_lv] = opL(dv_lv);
            [dptn_h_la, dptn_v_la] = opL(dv_la);
            [ptn_h, ptn_v] = opL(v);
            [Edptn_h_lv, Edptn_v_lv] = opL(Edv_lv);
            [Edptn_h_la, Edptn_v_la] = opL(Edv_la);
            [Eptn_h, Eptn_v] = opL(Ev);
            
            dt_h_lv = dt_h_lv + sig*dptn_h_lv; %prox argument
            dt_h_la = dt_h_la + sig*dptn_h_la; %prox argument
            t_h = t_h + sig*ptn_h; %prox argument
            Edt_h_lv = Edt_h_lv + sig*Edptn_h_lv; %prox argument
            Edt_h_la = Edt_h_la + sig*Edptn_h_la; %prox argument
            Et_h = Et_h + sig*Eptn_h; %prox argument
            
            dt_v_lv = dt_v_lv + sig*dptn_v_lv; %prox argument
            dt_v_la = dt_v_la + sig*dptn_v_la; %prox argument
            t_v = t_v + sig*ptn_v; %prox argument
            Edt_v_lv = Edt_v_lv + sig*Edptn_v_lv; %prox argument
            Edt_v_la = Edt_v_la + sig*Edptn_v_la; %prox argument
            Et_v = Et_v + sig*Eptn_v; %prox argument
            
            
            [dpgn_h_lv, dpgn_v_lv, dptn_h_lv, dptn_v_lv] = dprox_L12c_lambda(g_h/sig, g_v/sig, t_h/sig, t_v/sig, lambda/sig);
            [d1pgn_h_lv, d1pgn_v_lv, d1ptn_h_lv, d1ptn_v_lv] = dprox_L12c(g_h/sig, g_v/sig, t_h/sig, t_v/sig, dg_h_lv, dg_v_lv, dt_h_lv, dt_v_lv, lambda/sig);
            [d1pgn_h_la, d1pgn_v_la, d1ptn_h_la, d1ptn_v_la] = dprox_L12c(g_h/sig, g_v/sig, t_h/sig, t_v/sig, dg_h_la, dg_v_la, dt_h_la, dt_v_la, lambda/sig);
            [pgn_h, pgn_v, ptn_h, ptn_v] = prox_L12c(g_h/sig, g_v/sig, t_h/sig, t_v/sig, lambda/sig);
            [Edpgn_h_lv, Edpgn_v_lv, Edptn_h_lv, Edptn_v_lv] = dprox_L12c_lambda(Eg_h/sig, Eg_v/sig, Et_h/sig, Et_v/sig, lambda/sig);
            [Ed1pgn_h_lv, Ed1pgn_v_lv, Ed1ptn_h_lv, Ed1ptn_v_lv] = dprox_L12c(Eg_h/sig, Eg_v/sig, Et_h/sig, Et_v/sig, Edg_h_lv, Edg_v_lv, Edt_h_lv, Edt_v_lv, lambda/sig);
            [Ed1pgn_h_la, Ed1pgn_v_la, Ed1ptn_h_la, Ed1ptn_v_la] = dprox_L12c(Eg_h/sig, Eg_v/sig, Et_h/sig, Et_v/sig, Edg_h_la, Edg_v_la, Edt_h_la, Edt_v_la, lambda/sig);
            [Epgn_h, Epgn_v, Eptn_h, Eptn_v] = prox_L12c(Eg_h/sig, Eg_v/sig, Et_h/sig, Et_v/sig, lambda/sig);
            
            dg_h_lv=dg_h_lv - dpgn_h_lv - d1pgn_h_lv;
            dg_h_la=dg_h_la - d1pgn_h_la;
            g_h=g_h - sig*pgn_h;
            Edg_h_lv=Edg_h_lv - Edpgn_h_lv - Ed1pgn_h_lv;
            Edg_h_la=Edg_h_la - Ed1pgn_h_la;
            Eg_h=Eg_h - sig*Epgn_h;
            
            dg_v_lv=dg_v_lv - dpgn_v_lv - d1pgn_v_lv;
            dg_v_la=dg_v_la - d1pgn_v_la;
            g_v=g_v - sig*pgn_v;
            Edg_v_lv=Edg_v_lv - Edpgn_v_lv - Ed1pgn_v_lv;
            Edg_v_la=Edg_v_la - Ed1pgn_v_la;
            Eg_v=Eg_v - sig*Epgn_v;
            
            dt_h_lv=dt_h_lv - dptn_h_lv - d1ptn_h_lv;
            dt_h_la=dt_h_la - d1ptn_h_la;
            t_h=t_h - sig*ptn_h;
            Edt_h_lv=Edt_h_lv - Edptn_h_lv - Ed1ptn_h_lv;
            Edt_h_la=Edt_h_la - Ed1ptn_h_la;
            Et_h=Et_h - sig*Eptn_h;
            
            dt_v_lv=dt_v_lv - dptn_v_lv - d1ptn_v_lv;
            dt_v_la=dt_v_la - d1ptn_v_la;
            t_v=t_v - sig*ptn_v;
            Edt_v_lv=Edt_v_lv - Edptn_v_lv - Ed1ptn_v_lv;
            Edt_v_la=Edt_v_la - Ed1ptn_v_la;
            Et_v=Et_v - sig*Eptn_v;
            
            
            %Update of the descent steps
            theta = (1+2*mu_g*tau)^(-1/2);
            tau = theta*tau;
            sig=sig/theta;
            
            
            %Update dual auxiliary variable
            dbg_h_lv = dg_h_lv + theta*(dg_h_lv-dg_hs_lv);
            dbg_h_la = dg_h_la + theta*(dg_h_la-dg_hs_la);
            bg_h = g_h + theta*(g_h-g_hs);
            Edbg_h_lv = Edg_h_lv + theta*(Edg_h_lv-Edg_hs_lv);
            Edbg_h_la = Edg_h_la + theta*(Edg_h_la-Edg_hs_la);
            Ebg_h = Eg_h + theta*(Eg_h-Eg_hs);
            
            dbg_v_lv = dg_v_lv + theta*(dg_v_lv-dg_vs_lv);
            dbg_v_la = dg_v_la + theta*(dg_v_la-dg_vs_la);
            bg_v = g_v + theta*(g_v-g_vs);
            Edbg_v_lv = Edg_v_lv + theta*(Edg_v_lv-Edg_vs_lv);
            Edbg_v_la = Edg_v_la + theta*(Edg_v_la-Edg_vs_la);
            Ebg_v = Eg_v + theta*(Eg_v-Eg_vs);
            
            dbt_h_lv = dt_h_lv + theta*(dt_h_lv-dt_hs_lv);
            dbt_h_la = dt_h_la + theta*(dt_h_la-dt_hs_la);
            bt_h = t_h + theta*(t_h-t_hs);
            Edbt_h_lv = Edt_h_lv + theta*(Edt_h_lv-Edt_hs_lv);
            Edbt_h_la = Edt_h_la + theta*(Edt_h_la-Edt_hs_la);
            Ebt_h = Et_h + theta*(Et_h-Et_hs);
            
            dbt_v_lv = dt_v_lv + theta*(dt_v_lv-dt_vs_lv);
            dbt_v_la = dt_v_la + theta*(dt_v_la-dt_vs_la);
            bt_v = t_v + theta*(t_v-t_vs);
            Edbt_v_lv = Edt_v_lv + theta*(Edt_v_lv-Edt_vs_lv);
            Edbt_v_la = Edt_v_la + theta*(Edt_v_la-Edt_vs_la);
            Ebt_v = Et_v + theta*(Et_v-Et_vs);
            
            
            
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
    
    % Local regularity
    init_PD.dh_lv = dh_lv;
    init_PD.dh_la = dh_la;
    init_PD.h = h;
    init_PD.Edh_lv = Edh_lv;
    init_PD.Edh_la = Edh_la;
    init_PD.Eh = Eh;
    
    init_PD.dg_h_lv = dg_h_lv;
    init_PD.dg_v_lv = dg_v_lv;
    init_PD.dg_h_la = dg_h_la;
    init_PD.dg_v_la = dg_v_la;
    init_PD.g_h = g_h;
    init_PD.g_v = g_v;
    init_PD.Edg_h_lv = Edg_h_lv;
    init_PD.Edg_v_lv = Edg_v_lv;
    init_PD.Edg_h_la = Edg_h_la;
    init_PD.Edg_v_la = Edg_v_la;
    init_PD.Eg_h = Eg_h;
    init_PD.Eg_v = Eg_v;
    
    
    init_PD.dbg_h_lv = dbg_h_lv;
    init_PD.dbg_v_lv = dbg_v_lv;
    init_PD.dbg_h_la = dbg_h_la;
    init_PD.dbg_v_la = dbg_v_la;
    init_PD.bg_h = bg_h;
    init_PD.bg_v = bg_v;
    init_PD.Edbg_h_lv = Edbg_h_lv;
    init_PD.Edbg_v_lv = Edbg_v_lv;
    init_PD.Edbg_h_la = Edbg_h_la;
    init_PD.Edbg_v_la = Edbg_v_la;
    init_PD.Ebg_h = Ebg_h;
    init_PD.Ebg_v = Ebg_v;
    
    
    
    % Local variance
    init_PD.dv_lv = dv_lv;
    init_PD.dv_la = dv_la;
    init_PD.v = v;
    init_PD.Edv_lv = Edv_lv;
    init_PD.Edv_la = Edv_la;
    init_PD.Ev = Ev;
    
    init_PD.dt_h_lv = dt_h_lv;
    init_PD.dt_v_lv = dt_v_lv;
    init_PD.dt_h_la = dt_h_la;
    init_PD.dt_v_la = dt_v_la;
    init_PD.t_h = t_h;
    init_PD.t_v = t_v;
    init_PD.Edt_h_lv = Edt_h_lv;
    init_PD.Edt_v_lv = Edt_v_lv;
    init_PD.Edt_h_la = Edt_h_la;
    init_PD.Edt_v_la = Edt_v_la;
    init_PD.Et_h = Et_h;
    init_PD.Et_v = Et_v;
    
    
    init_PD.dbt_h_lv = dbt_h_lv;
    init_PD.dbt_v_lv = dbt_v_lv;
    init_PD.dbt_h_la = dbt_h_la;
    init_PD.dbt_v_la = dbt_v_la;
    init_PD.bt_h = bt_h;
    init_PD.bt_v = bt_v;
    init_PD.Edbt_h_lv = Edbt_h_lv;
    init_PD.Edbt_v_lv = Edbt_v_lv;
    init_PD.Edbt_h_la = Edbt_h_la;
    init_PD.Edbt_v_la = Edbt_v_la;
    init_PD.Ebt_h = Ebt_h;
    init_PD.Ebt_v = Ebt_v;
    
    set_init(init_PD);
end