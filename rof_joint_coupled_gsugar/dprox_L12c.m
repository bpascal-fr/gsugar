function [dph, dpv, dqh, dqv] = dprox_L12c(yh,yv,zh, zv, dyh,dyv, dzh, dzv, lambda)
    
    % Compute the differential of the proximal operator of l_2,1 norm
    %       (xh,xv,uh,uv) --> lambda * sum( sqrt(xh^2 + xv^2 + uh^2 + uv^2) )
    % with respect to variable (xh,xv,uh,uv), at point (yh,yv,zh,zv) applied to (dyh,dyv,dzh,dzv) whose explicit expression is
    %       (xh,xv,uh,uv,dxh,dxv,duh,duv) --> | (dxh,dxv,duh,duv) - lambda/|(xh,xv,uh,uv)|_{2,1} ((dxh,dxv,duh,duv) - <(dxh,dxv,duh,duv),(xh,xv,uh,uv)>(xh,xv,uh,uv) /|(xh,xv,uh,uv)|_{2,1}^2) if |(xh,xv,uh,uv)|_{2,1} > lambda
    %                                         |        0    else
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
    nyz = (yh.^2+yv.^2+zh.^2+zv.^2).^(1/2);
    ind = find(nyz >lambda);
    
    
    dph = zeros(size(yh));
    dpv = zeros(size(yv));
    dqh = zeros(size(yh));
    dqv = zeros(size(yv));
    
    tmp_proj = (dyh(ind).*yh(ind) + dyv(ind).*yv(ind) + dzh(ind).*zh(ind) + dzv(ind).*zv(ind))./nyz(ind).^2;
    projyh = dyh(ind) -   tmp_proj.*yh(ind);
    projyv = dyv(ind) -   tmp_proj.*yv(ind);
    projzh = dzh(ind) -   tmp_proj.*zh(ind);
    projzv = dzv(ind) -   tmp_proj.*zv(ind);
    
    dph(ind) = dyh(ind) - lambda./nyz(ind).*projyh;
    dpv(ind) = dyv(ind) - lambda./nyz(ind).*projyv;
    dqh(ind) = dzh(ind) - lambda./nyz(ind).*projzh;
    dqv(ind) = dzv(ind) - lambda./nyz(ind).*projzv;
    
end