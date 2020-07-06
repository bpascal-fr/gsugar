function [dph, dpv, dqh, dqv] = dprox_L12c_lambda(yh,yv,zh,zv,lambda)
    
    % Compute the differential of the proximal operator of l_2,1 norm
    %       (xh,xv,uh,uv) --> lambda * sum( sqrt(xh^2 + xv^2 + uh^2 + uv^2) )
    % at point (yh,yv,zh,zv) with respect to parameter lambda whose
    % explicit expression is
    %       (xh,xv,uh,uv) --> | (- xh/|(xh,xv,uh,uv)|_{2,1},- xv/|(xh,xv,uh,uv)|_{2,1},- uh/|(xh,xv,uh,uv)|_{2,1},- uv/|(xh,xv,uh,uv)|_{2,1}) if
    %       |(xh,xv,uh,uv)|_{2,1} > lambda
    %                 |             0                             else
    %
    % inputs  - (yh,yv,zh,zv): current point
    %         - (dyh,dyv,dzh,dzv): vector on which the differential is applied
    %         - lambda: multiplicative factor
    %
    % output  - (dph, dpv, dqh, dqv): differential of proximal operator with respect to
    % variable (yh,yv,zh,zv) applied to (dyh,dyv,dzh,dzv)
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    nyz = (yh.^2+yv.^2+zh.^2+zv.^2).^(1/2);
    ind = find(nyz >lambda);
    
    dph = zeros(size(yh));
    dpv = zeros(size(yv));
    dqh = zeros(size(yh));
    dqv = zeros(size(yv));
    
    dph(ind) = - yh(ind)./nyz(ind);
    dpv(ind) = - yv(ind)./nyz(ind);
    dqh(ind) = - zh(ind)./nyz(ind);
    dqv(ind) = - zv(ind)./nyz(ind);
    
end