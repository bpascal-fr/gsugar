function display_result_1D(x,Th)
    
    h = x.h;
    
    meth = x.meth;
    
    
    figure(1);
    if strcmp(meth,'ROF')
        subplot(234);plot(1:length(h),h,'linewidth',2); grid on; hold on; plot(Th,'linewidth',2)
        set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on
        title(['$\widehat{h}^{\mathrm{',meth,'}}$'],'interpreter','latex','fontsize',20)
    elseif strcmp(meth,'J')
        subplot(235);plot(1:length(h),h,'linewidth',2); grid on; hold on; plot(Th,'linewidth',2)
        set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on
        title(['$\widehat{h}^{\mathrm{',meth,'}}$'],'interpreter','latex','fontsize',20)
    else
        subplot(236);plot(1:length(h),h,'linewidth',2); grid on; hold on; plot(Th,'linewidth',2)
        set(gca,'fontsize',20,'ticklabelinterpreter','latex'); grid on
        title(['$\widehat{h}^{\mathrm{',meth,'}}$'],'interpreter','latex','fontsize',20)
    end
    
    
end