function [p,q] = prox_L1c(y,z,lambda)
    
    % Compute the proximal operator of l_2,1 norm
    %       (x,u) --> lambda * sum( sqrt(x^2 + u^2) )
    % at point (y,z)
    %
    % inputs  - (y,z): current point
    %         - lambda: multiplicative factor
    %
    % output  - (p,q): proximal point
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    nyz = (y.^2+z.^2).^(1/2);
    ind = find(nyz >lambda);
    
    p = zeros(size(y));
    q = zeros(size(z));
    
    p(ind) = (1 - lambda./nyz(ind)).*y(ind);
    q(ind) = (1 - lambda./nyz(ind)).*z(ind);
    
end