% Function computing SURE and SUGAR for Rudin-Osher-Fatemi problem performing 
% Total Variation based denoising of linear regression estimate of local 
% regularity, with Finite Difference Monte Carlo
% for given log2-leaders L and hyperparameter lbd using the covariance 
% matrix S with finite difference step sure.eps and Monte Carlo vector
% sure.delta (image)
%
% from 
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246
% and
% - B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [sure_fdmc,sugar,h, Eh, dh, Edh] = SURE_SUGAR_rof_GRANSO(L, lbd, sure, S)
    
    
    % inputs   - L: log2-leaders of the image to be segmented
    %          - lbd: regularization parameter
    %          - sure: delta and and epsilon for Monte Carlo and Finite Difference resp.
    %          - S: covariance matrix of the noise
    %
    % ouputs   - sure_fdmc: estimation of the risk with SURE FDMC
    %          - sugar: estimation of the gradient of the risk with SUGAR FDMC
    %          - h: regularized estimate of local regularity
    %          - Eh: perturbed regularized estimate of local regularity
    %          - dh: gradient of regularized estimate of local regularity
    %          w.r.t. regularization parameter
    %          - Edh: gradient of perturbed regularized estimate of local regularity
    %          w.r.t. regularization parameter
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    prox.lambda = lbd;
    [h, Eh, dh, Edh]=dPA_PD(L,prox,sure);
    
    tmp = get_init;
    if isfield(tmp,'h')
        figure(1);
        subplot(234);imagesc(tmp.h); axis off image; colormap(gray)
        title('$\widehat{h}^{\mathrm{ROF}}$','interpreter','latex','fontsize',20)
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
    Cdelta = opC(sure.delta, S, JJ);
    [~, hd] = linear_reg(Cdelta,JJ);
    dof_fdmc = sum((Eh(:)-h(:))/sure.eps.*hd(:));
    sure_fdmc = norm(h - L.h_LR,'fro')^2 - trAadjAC + 2*dof_fdmc;
    
    
    
    % Compute SUGAR Finite Difference Monte Carlo
    sugar = 2*sum(sum(dh.*(h - L.h_LR))) + 2*sum(sum((Edh - dh).*hd))/sure.eps;
    
    
end
