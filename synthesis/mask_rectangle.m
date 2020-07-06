function [H, S, MASK] = mask_rectangle(N,H0,H1,S0,S1)
    
    H = H0*ones(2^N,2^N); 
    S = S0*ones(2^N,2^N);
    MASK = ones(2^N,2^N);
    x1 = linspace(-5,5,2^N);
    x2 = x1;
    [X1, X2] = meshgrid(x1,x2);
    
    S((X1<2.5)&(X1>-2.5)&(X2<2.5)&(X2>-2.5)) = S1;
    H((X1<2.5)&(X1>-2.5)&(X2<2.5)&(X2>-2.5)) = H1;
    MASK((X1<2.5)&(X1>-2.5)&(X2<2.5)&(X2>-2.5)) = 2;
    
end