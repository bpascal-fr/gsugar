function dp = dprox_L1(y,dy,lambda)
    
    % Compute the differential of proximal operator of l1 norm
    %       x --> lambda * ||x||_1
    % with respect to variable x, at point y applied to dy whose explicit expression
    % is
    %       (x,dx) --> | dx - lambda/|x| (dx - <dx,x>x /|x|^2) if |x| > lambda
    %                  |        0    else
    %
    % inputs  - y: current point
    %         - dy: vector on which the differential is applied
    %         - lambda: multiplicative factor
    %
    % output  - dp: differential of proximal operator with respect to
    % variable y applied to vector dy
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    ny = abs(y);
    ind = find(ny >lambda);
    
    
    dp = zeros(size(dy));
    
    tmp_proj = (dy(ind).*y(ind))./ny(ind).^2;
    projy = dy(ind) -   tmp_proj.*y(ind);
    
    dp(ind) = dy(ind) - lambda./ny(ind).*projy;
    
end