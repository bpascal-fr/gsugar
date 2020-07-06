function [dph, dpv] = dprox_L12_lambda(yh,yv,lambda)
    
    % Compute the differential of the proximal operator of l_2,1 norm
    %       (xh,xv) --> lambda * sum( sqrt(xh^2 + xv^2) )
    % at point (yh,yv) with respect to parameter lambda whose explicit
    % expression is
    %        (xh,xv) --> | (- xh/|(xh,xv)|_{2,1},- xv/|(xh,xv)|_{2,1}) if
    %       |(xh,xv)|_{2,1} > lambda
    %                 |             0                             else
    %
    %
    % inputs  - (yh,yv): current point
    %         - lambda: multiplicative factor
    %
    % output  - (dph,dpv): differential of proximal operator with respect to
    % lambda at point (yh,yv)
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    
    ny = (yh.^2+yv.^2).^(1/2);
    ind = find(ny >lambda);
    
    
    dph = zeros(size(yh));
    dpv = zeros(size(yv));
    dph(ind) = -yh(ind)./ny(ind);
    dpv(ind) = -yv(ind)./ny(ind);
    
end