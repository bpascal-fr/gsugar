function [dp, dq] = dprox_L1c_lambda(y,z,lambda)
    
    % Compute the differential of proximal operator of l_2,1 norm
    %       (x,u) --> lambda * sum( sqrt(x^2 + u^2) )
    % at point (y,z) with respect to parameter lambda whose explicit
    % expression is
    %       (x,u) --> | (- x/|(x,u)|_{2,1},- u/|(x,u)|_{2,1}) if
    %       |(x,u)|_{2,1} > lambda
    %                 |             0                             else
    %
    % inputs  - (y,z): current point
    %         - lambda: multiplicative factor
    %
    % output  - (dp,dq): differential of proximal operator with respect to
    % lambda at point (y,z)
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    nyz = (y.^2+z.^2).^(1/2);
    ind = find(nyz >lambda);
    
    dp = zeros(size(y));
    dq = zeros(size(z));
    
    dp(ind) = - y(ind)./nyz(ind);
    dq(ind) = - z(ind)./nyz(ind);
    
end