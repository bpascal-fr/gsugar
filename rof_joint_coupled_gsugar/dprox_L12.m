function [dph, dpv] = dprox_L12(yh,yv,dyh,dyv,lambda)
    
    % Compute the differential of proximal operator of l_2,1 norm
    %       (xh,xv) --> lambda * sum( sqrt(xh^2 + xv^2) )
    % with respect to variable (xh,xv), at point (yh,yv) applied to (dyh,dyv) whose explicit expression
    % is
    %       (xh,xv,dxh,dxv) --> | (dxh,dxv) - lambda/|(xh,xv)|_{2,1} ((dxh,dxv) - <(dxh,dxv),(xh,xv)>(xh,xv) /|(xh,xv)|_{2,1}^2) if |(xh,xv)|_{2,1} > lambda
    %                           |        0    else
    %
    % inputs  - (yh,yv): current point
    %         - (dyh,dyv): vector on which the differential is applied
    %         - lambda: multiplicative factor
    %
    % output  - (dph,dpv): differential of proximal operator with respect to
    % variable (yh,yv) applied to vector (dyh,dyv)
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    ny = (yh.^2+yv.^2).^(1/2);
    ind = find(ny >lambda);
    
    
    dph = zeros(size(dyh));
    dpv = zeros(size(dyv));
    
    tmp_proj = (dyh(ind).*yh(ind) + dyv(ind).*yv(ind))./ny(ind).^2;
    projyh = dyh(ind) -   tmp_proj.*yh(ind);
    projyv = dyv(ind) -   tmp_proj.*yv(ind);
    
    dph(ind) = dyh(ind) - lambda./ny(ind).*projyh;
    dpv(ind) = dyv(ind) - lambda./ny(ind).*projyv;
    
    
    
end