function [ph, pv] = prox_L12(yh,yv,lambda)
    
    % Compute the proximal operator of l_2,1 norm
    %       (xh,xv) --> lambda * sum( sqrt(xh^2 + xv^2) )
    % at point (yh,yv)
    %
    % inputs  - (yh,yv): current point
    %         - lambda: multiplicative factor
    %
    % output  - (ph,pv): proximal point
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    ny = (yh.^2+yv.^2).^(1/2);
    ind = find(ny >lambda);
    
    ph = zeros(size(yh));
    pv = zeros(size(yv));
    
    ph(ind) = (1 - lambda./ny(ind)).*yh(ind);
    pv(ind) = (1 - lambda./ny(ind)).*yv(ind);
    
end