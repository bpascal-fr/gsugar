function [dp, dq] = dprox_L1c(y,z, dy, dz, lambda)
    
    % Compute the differential of proximal operator of l_2,1 norm
    %       (x,u) --> lambda * sum( sqrt(x^2 + u^2) )
    % with respect to variable (x,u), at point (y,z) applied to (dy,dz) whose explicit expression
    % is
    %       (x,u,dx,du) --> | (dx,du) - lambda/|(x,u)|_{2,1} ((dx,du) - <(dx,du),(x,u)>(x,u) /|(x,u)|_{2,1}^2) if |(x,u)|_{2,1} > lambda
    %                       |        0    else
    %
    % inputs  - (y,z): current point
    %         - (dy,dz): vector on which the differential is applied
    %         - lambda: multiplicative factor
    %
    % output  - (dp,dq): differential of proximal operator with respect to
    % variable (y,z) applied to vector (dy,dz)
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    nyz = (y.^2+z.^2).^(1/2);
    ind = find(nyz >lambda);
    
    
    dp = zeros(size(y));
    dq = zeros(size(z));
    
    tmp_proj = (dy(ind).*y(ind) + dz(ind).*z(ind))./nyz(ind).^2;
    projy = dy(ind) -   tmp_proj.*y(ind);
    projz = dz(ind) -   tmp_proj.*z(ind);
    
    dp(ind) = dy(ind) - lambda./nyz(ind).*projy;
    dq(ind) = dz(ind) - lambda./nyz(ind).*projz;
    
end