function [ph, pv, qh, qv] = prox_L12c(yh,yv,zh,zv,lambda)
    
    % Compute the proximal operator of l_2,1 norm
    %       (xh,xv,uh,uv) --> lambda * sum( sqrt(xh^2 + xv^2 + uh^2 + uv^2) )
    % at point (yh,yv,zh,zv)
    %
    % inputs  - (yh,yv,zh,zv): current point
    %         - lambda: multiplicative factor
    %
    % output  - (ph, pv, qh, qv): proximal point
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    nyz = (yh.^2+yv.^2+zh.^2+zv.^2).^(1/2);
    ind = find(nyz >lambda);
    
    ph = zeros(size(yh));
    pv = zeros(size(yv));
    
    qh = zeros(size(yh));
    qv = zeros(size(yv));
    
    ph(ind) = (1 - lambda./nyz(ind)).*yh(ind);
    pv(ind) = (1 - lambda./nyz(ind)).*yv(ind);
    
    qh(ind) = (1 - lambda./nyz(ind)).*zh(ind);
    qv(ind) = (1 - lambda./nyz(ind)).*zv(ind);
    
end