function [H, S, MASK] = mask_ellipse(N,H0,H1,S0,S1)
    
    
    
    %% CREATION OF MASKS

    MASK = ones(2^N,2^N);
    H = H0*ones(2^N,2^N);
    S = S0*ones(2^N,2^N);
    x = linspace(-20,20,2^N);
    y = x;
    [X,Y] = meshgrid(x,y);
    
    MASK(sqrt(3)*X.^2+5*Y.^2 < 500) = 2;
    H(sqrt(3)*X.^2+5*Y.^2 < 500) = H1;
    S(sqrt(3)*X.^2+5*Y.^2 < 500) = S1;
    
   
end