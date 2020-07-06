function p = prox_L1(y,lambda)
    
    % Compute the proximal operator of l1 norm
    %       x --> lambda * ||x||_1
    % at point y
    %
    % inputs  - y: current point
    %         - lambda: multiplicative factor
    %
    % output  - p: proximal point
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    ny = abs(y);
    ind = find(ny >lambda);
    
    p = zeros(size(y));
    
    p(ind) = (1 - lambda./ny(ind)).*y(ind);
    
end