% Function computing SURE and SUGAR for joint functional for fractal texture 
% segmentation, with Finite Difference Monte Carlo
% for given log2-leaders L and hyperparameters lbd using the covariance 
% matrix S with finite difference step sure.eps and Monte Carlo vector
% sure.delta (image)
%
% from B. Pascal, N. Pustelnik, P. Abry: How Joint Fractal Features Estimation 
% and Texture Segmentation can be cast into a Strongly Convex Optimization
% Problem ?. arxiv:1910.05246
% and B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [sure_fdmc,sugar,v,h,Eh,dh_lv, dh_lh,Edh_lv,Edh_lh] = SURE_SUGAR_joint_GRANSO(L, lbd, sure, S)
    
    % inputs   - L: log2-leaders of the image to be segmented
    %          - lbd: regularization parameters
    %          - sure: delta and and epsilon for Monte Carlo and Finite Difference resp.
    %          - S: covariance matrix of the noise
    %
    % ouputs   - sure_fdmc: estimation of the risk with SURE FDMC
    %          - sugar: estimation of the gradient of the risk with SUGAR FDMC
    %          - v: regularized estimate of local power
    %          - h: regularized estimate of local regularity
    %          - Ev: perturbed regularized estimate of local power
    %          - Eh: perturbed regularized estimate of local regularity
    %          - dv_lv: gradient of regularized estimate of local power
    %          w.r.t. regularization parameter lambda_v
    %          - dv_lh: gradient of regularized estimate of local power
    %          w.r.t. regularization parameter lambda_h
    %          - dh_lv: gradient of regularized estimate of local regularity
    %          w.r.t. regularization parameter lambda_v
    %          - dh_lh: gradient of regularized estimate of local regularity
    %          w.r.t. regularization parameter lambda_h
    %          - Edv_lv: gradient of perturbed regularized estimate of local power
    %          w.r.t. regularization parameter lambda_v
    %          - Edv_lh: gradient of perturbed regularized estimate of local power
    %          w.r.t. regularization parameter lambda_h
    %          - Edh_lv: gradient of perturbed regularized estimate of local regularity
    %          w.r.t. regularization parameter lambda_v
    %          - Edh_lh: gradient of perturbed regularized estimate of local regularity
    %          w.r.t. regularization parameter lambda_h
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    prox.lambda_v= lbd(1);
    prox.lambda_h= lbd(2);
    [v, h, ~, Eh, ~, ~, dh_lv, dh_lh, ~, ~, Edh_lv, Edh_lh]=dPA_PDj(L,prox,sure);
    
    tmp = get_init;
    if isfield(tmp,'h')
        figure(1);
        subplot(235);imagesc(tmp.h); axis off image
        title('$\widehat{h}^{\mathrm{J}}$','interpreter','latex','fontsize',20)
        pause(0.1)
    end
    
    % Compute tr(A*Pi*PiAC) for non scalar covariance matrix
    JJ = L.JJ; J2 = JJ(end);
    JJ_tmp = zeros(1,J2); JJ_tmp(JJ) = JJ;
    S0=sum(JJ.^0); S1=sum(JJ); S2=sum(JJ.^2); det = S2*S0 - S1*S1; DJJ = diag(JJ_tmp);
    cov_synth = zeros(J2,J2);
    for ii = JJ
        for jj = JJ
            cov_synth(ii,jj) = max(S{ii,jj}(:));
        end
    end
    trAadjAC = numel(h)*(S1^2*sum(sum(cov_synth)) - 2*sum(sum(DJJ*cov_synth))*S0*S1 + S0^2*sum(sum(DJJ*cov_synth*DJJ)))/det^2;
    
    
    
    
    % Compute SURE Finite Difference Monte Carlo
    phi_X = opA(v,h,JJ);
    tmp_Y = cell(length(JJ));
    for jj = JJ
        tmp_Y{jj} = phi_X{jj} - log2(L.leaders{jj});
    end
    Cdelta = opC(sure.delta, S, JJ);
    [~, hd] = linear_reg(Cdelta,JJ);
    [~, hy] = linear_reg(tmp_Y,JJ);
    dof_fdmc = sum((Eh(:)-h(:))/sure.eps.*hd(:));
    sure_fdmc = norm(hy,'fro')^2 - trAadjAC + 2*dof_fdmc;
    
    
    
    % Compute SUGAR Finite Difference Monte Carlo
    sugar(1) = 2*sum(sum(dh_lv.*hy)) + 2*sum(sum((Edh_lv - dh_lv).*hd))/sure.eps;
    sugar(2) = 2*sum(sum(dh_lh.*hy)) + 2*sum(sum((Edh_lh - dh_lh).*hd))/sure.eps;
    
    sugar = sugar';
    
    
    
end
