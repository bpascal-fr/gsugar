% Evaluation of the objective function of primal and dual functional.
% and derivation of the duality gap of the coupled functional for fractal
% texture segmentation. (image)
%
% from B. Pascal, N. Pustelnik, P. Abry: How Joint Fractal Features Estimation 
% and Texture Segmentation can be cast into a Strongly Convex Optimization
% Problem ?. arxiv:1910.05246


function [crit, dual, gap] = PDc_gap(sn, hn, tn_h, tn_v, gn_h, gn_v, L, adj_ts, adj_th, f_adj0, JJ, prox, use)
    
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
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    lambda = prox.lambda;
    alpha = prox.alpha;
    
    %% Compute primal criterion
    f=0; tv=0;
    for jj = JJ
        f = f + norm(sn + jj* hn - L{jj},'fro')^2;
    end
    [sjh,sjv] = opL(sn);
    [hjh,hjv] = opL(hn);
    tv = tv + lambda*sum(sum((sjh.^2 + sjv.^2 + alpha^2*hjh.^2 + alpha^2*hjv.^2).^(1/2)));
    
    
    crit = 1/2 * f + tv;
    
    %% Compute dual criterion
    %Data fidelity conjugate
    
    szn = opL_adj(tn_h, tn_v);
    hzn = opL_adj(gn_h, gn_v);
    
    Aadj_szn = (use.S2*szn - use.S1*alpha*hzn)/use.det;
    Aadj_hzn = (use.S0*alpha*hzn - use.S1*szn)/use.det;
    
    dual = 1/2*(sum(szn(:).*Aadj_szn(:) + alpha*hzn(:).*Aadj_hzn(:))) - (sum(Aadj_szn(:).*adj_ts(:) + Aadj_hzn(:).*adj_th(:))) + f_adj0;
    
    %Distance to the lambda radius ball for 2,1 norm
    [p1, p2, p3, p4] = prox_L12c(tn_h,tn_v,gn_h, gn_v, lambda);
    norm_m = norm([p1, p2, p3, p4] , 'fro');
    dual = dual + norm_m;
    
    %% Compute dual gap
    gap = crit + dual;
end