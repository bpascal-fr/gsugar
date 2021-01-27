function display_result(x,seg)
    
    contour_color = [0.7294,0.0392,0.0392];
    
    h = x.h;
    
    % Contour detection
    [Hs,Vs] = opL(seg);
    Ds      = sqrt(Hs.^2+Vs.^2);
    deps    = max(1,sqrt(numel(h))/2^7); % width of the contour
    bDs     = conv2(Ds,ones(deps,deps)/deps,'same'); % contour spreading
    binDs   = bDs > 0;
    
    % Normalization of estimate for correct display
    normh   = (h - min(h(:)))/(max(h(:)) - min(h(:)));
    
    % Grayscale estimate and red contours
    SEG1 = normh; SEG1(binDs) = contour_color(1);
    SEG2 = SEG1; SEG2(binDs) = contour_color(2);
    SEG3 = SEG1; SEG3(binDs) = contour_color(3);
    
    seg_contour(:,:,1) = SEG1;
    seg_contour(:,:,2) = SEG2;
    seg_contour(:,:,3) = SEG3;
    
    
    meth = x.meth;
    
    figure(1);
    if strcmp(meth,'ROF')
        subplot(234);imagesc(seg_contour); axis off image
        title(['$\widehat{h}^{\mathrm{',meth,'}}$'],'interpreter','latex','fontsize',20)
    elseif strcmp(meth,'J')
        subplot(235);imagesc(seg_contour); axis off image
        title(['$\widehat{h}^{\mathrm{',meth,'}}$'],'interpreter','latex','fontsize',20)
    else
        subplot(236);imagesc(seg_contour); axis off image
        title(['$\widehat{h}^{\mathrm{',meth,'}}$'],'interpreter','latex','fontsize',20)
    end
end