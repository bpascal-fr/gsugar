
function U=fbm1D(N,H,Variance)
    
    
    % Generate homogeneous fractal process with prescribed
    % local regularity and local variance from a standard model of
    % Fractional Brownian Motion
    %
    % from
    % - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
    % Joint Fractal Feature Estimation and Texture Segmentation, 
    % (2019) arxiv:1910.05246
    % 
    % Makes use of the implementation of FBM by M. Clausel and B. Vedel, 
    % 2008, updated by S. Roux, 2011
    
    %% Modified by B. Pascal
    
    % July, 18th 2018
    l1=1;l2=1;M=N-1;
    
    %COORDINATES
    X=(-2*2^M:2:2*2^M)/(2^(M+1));
    X(2^M+1)=1/2^M;
    
    %NOUVELLE DENSITE SPECTRALE PSI
    tau=abs(X);
    phi=tau.^(H+(l1)/2);
    
    
    
    %CONSTRUCTION OF THE FOURIER TRANSFORM OF THE FIELD (WITHOUT RENORMALIZATION)
    %W= Fourier transform of the OSSRGF
    %Zr and Zi avoid to include additional symmetries in Fourier
    Zr=randn(1,2*2^M+1);
    Zi=randn(1,2*2^M+1);
    Z = Zr + 1i*Zi;
    
    
    W=fftshift(fft2(Z))./phi;

    
    %CONSTRUCTION OF THE FIELD (FOURIER INVERSE + RENORMALIZATION)
    T=real(ifft2(ifftshift(W)));
    Zp=T-T(2^M+1);
    Zp = Zp/std(Zp);
    U = sqrt(Variance).*Zp;
    
end
