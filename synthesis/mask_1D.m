function [H, S, MASK] = mask_1D(N,H0,H1,S0,S1)
    
    H = H0*ones(1,2^N); 
    S = S0*ones(1,2^N);
    MASK = ones(1,2^N);
    X = linspace(-5,5,2^N);
    
    S((X<1.75)&(X>-1.75)) = S1;
    H((X<1.75)&(X>-1.75)) = H1;
    MASK((X<1.75)&(X>-1.75)) = 2;
    
end