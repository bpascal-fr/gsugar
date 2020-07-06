function dp = dprox_L1_lambda(y,lambda)
    
    % Compute the differential of proximal operator of l1 norm
    %       x --> lambda * ||x||_1
    % at point y with respect to parameter lambda whose explicit expression
    % is
    %       x --> | - x/|x| if |x| > lambda
    %             |    0    else 
    %
    % inputs  - y: current point
    %         - lambda: multiplicative factor
    %
    % output  - dp: differential of proximal operator with respect to
    % lambda at point y
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    ny = abs(y);
    ind = find(ny >lambda);
    
    
    dp = zeros(size(y));
    dp(ind) = -y(ind)./ny(ind);
end