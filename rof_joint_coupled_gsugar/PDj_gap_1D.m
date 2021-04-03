% Evaluation of the objective function of primal and dual functional.
% and derivation of the duality gap of the joint functional for fractal
% process segmentation. (signal)
%
% from
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246

function [crit, dual, gap] = PDj_gap_1D(sn, hn, tn, gn, L, adj_ts, adj_th, f_adj0, JJ, prox, use)
    
    % inputs  - sn, hn: primal variables
    %         - tn_h, tn_v, gn_h, gn_v: dual variables
    %         - L.leaders: undecimated wavelet leaders
    %         - adj_ts, adj_th, f_adj0: auxiliary quantities not varying
    %         from one iteration to another
    %         - JJ: range of considered scales
    %         - prox: (lambda_v, lambda_h) regularization parameters
    %         - use: contains the needed quantities to perform linear
    %         regression
    %
    % outputs - crit: value of primal criterion
    %         - dual: value of minus the dual criterion
    %         - gap: duality gap
    
    lambda_v = prox.lambda_v;
    lambda_h = prox.lambda_h;
    
    %% Compute primal criterion
    f=0; tv=0;
    for jj = JJ
        f = f + norm(sn + jj* hn - L{jj},'fro')^2;
    end
    sj = opL_1D(sn);
    hj = opL_1D(hn);
    tv = tv + lambda_v*sum(abs(sj)) + lambda_h*sum(abs(hj));
    
    crit = 1/2 * f + tv;
    
    %% Compute dual criterion
    %Data fidelity conjugate
    
    szn = opL_adj_1D(tn);
    hzn = opL_adj_1D(gn);
    
    Aadj_szn = (use.S2*szn - use.S1*hzn)/use.det;
    Aadj_hzn = (use.S0*hzn - use.S1*szn)/use.det;
    
    dual = 1/2*(sum(szn.*Aadj_szn + hzn.*Aadj_hzn)) - (sum(Aadj_szn.*adj_ts + Aadj_hzn.*adj_th)) + f_adj0;
    
    %Distance to the lambda radius ball for 2,1 norm
    ptn = prox_L1(tn,lambda_v);
    pgn = prox_L1(gn,lambda_h);
    norm_m = norm([ptn, pgn] , 'fro');
    dual = dual + norm_m;
    
    %% Compute dual gap
    gap = crit + dual;
    
end