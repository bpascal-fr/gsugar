function z=prox_g(y, T,tau)
    
    % Compute proximal operator of x --> tau/2 || x - T ||^2 at point y
    %
    % input  - y: current point
    %        - T: observation
    %        - tau: multiplicative factor
    %
    % output - z: proximal point
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    temp=tau+1;
    z=tau*T/temp+y/temp;
    
end