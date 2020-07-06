% Function computing SURE and SUGAR for Total Variation based denoising (signal) 
% 
% from B. Pascal, S. Vaiter, N. Pustelnik, P. Abry: Automated data-driven 
% selection of the hyperparameters for Total-Variation based texture
% segmentation, (2020) arxiv:2004.09434

function [sure_fdmc,sugar,signal, Esignal, dsignal, Edsignal] = SURE_SUGAR_tv_GRANSO_1D(X, lbd, sure, S)
    
    % inputs  - X.signal_n: observed signal to be denoised
    %         - lbd: regularization parameter
    %         - sure: delta (Monte Carlo vector) eps (Finite Difference step)
    %         - S: estimated covariance structure
    %
    % ouputs  - sure_fdmc: estimation of the risk with SURE FDMC
    %         - sugar: estimation of the gradient of the risk with SUGAR FDMC
    %         - signal: denoised signal
    %         - Esignal: perturbed denoised signal
    %         - dsignal: gradient of denoised signal
    %         w.r.t. regularization parameter
    %         - Edsignal: gradient of perturbed denoised signal
    %         w.r.t. regularization parameter
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    prox.lambda = lbd;
    [signal, Esignal, dsignal, Edsignal] = dPA_PDtv_1D(X,prox,sure);
    
    tmp = get_init;
    if isfield(tmp,'x')
        figure(1);
        subplot(133);plot(1:length(tmp.h),tmp.h,'linewidth',2); title('Noisy signal','interpreter','latex','fontsize',12); set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on
        title('$\widehat{x}^{\mathrm{TV}}$','interpreter','latex','fontsize',20)
        pause(0.1)
    end
    
    
    % Compute SURE Finite Difference Monte Carlo
    Cdelta = conv(sure.delta,S,'same');
    dof_fdmc = sum((Esignal(:)-signal(:))/sure.eps.*Cdelta(:));
    sure_fdmc = norm(signal - X.signal_n,'fro')^2 + 2*dof_fdmc - X.D*numel(X.signal_n);
    
    
    
    % Compute SUGAR Finite Difference Monte Carlo
    sugar = 2*sum(sum(dsignal.*(signal - X.signal_n))) + 2*sum(sum((Edsignal - dsignal).*Cdelta))/sure.eps;
    
end
