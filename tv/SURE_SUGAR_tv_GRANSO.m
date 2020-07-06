% Function computing SURE and SUGAR for Total Variation based denoising (image) 
% 
% from B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [sure_fdmc,sugar,img, Eimg, dimg, Edimg] = SURE_SUGAR_tv_GRANSO(I, lbd, sure, S)
    
    % inputs  - I.img_n: observed image to be denoised
    %         - lbd: regularization parameter
    %         - sure: delta (Monte Carlo vector) eps (Finite Difference step)
    %         - S: estimated covariance structure
    %
    % ouputs  - sure_fdmc: estimation of the risk with SURE FDMC
    %         - sugar: estimation of the gradient of the risk with SUGAR FDMC
    %         - img: denoised image
    %         - Eimg: perturbed denoised image
    %         - dimg: gradient of denoised image
    %         w.r.t. regularization parameter
    %         - Edimg: gradient of perturbed denoised image
    %         w.r.t. regularization parameter
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    
    prox.lambda = lbd;
    [img, Eimg, dimg, Edimg] = dPA_PDtv(I,prox,sure);
    
    tmp = get_init;
    if isfield(tmp,'x')
        figure(1);
        subplot(133);imagesc(tmp.x); axis off image
        title('$\widehat{x}^{\mathrm{TV}}$','interpreter','latex','fontsize',20)
        pause(0.1)
    end
    
    
    % Compute SURE Finite Difference Monte Carlo
    Cdelta = conv2(sure.delta,S,'same');
    dof_fdmc = sum((Eimg(:)-img(:))/sure.eps.*Cdelta(:));
    sure_fdmc = norm(img - I.img_n,'fro')^2 + 2*dof_fdmc - I.D*numel(I.img_n);
    
    
    
    % Compute SUGAR Finite Difference Monte Carlo
    sugar = 2*sum(sum(dimg.*(img - I.img_n))) + 2*sum(sum((Edimg - dimg).*Cdelta))/sure.eps;
    
    
end
