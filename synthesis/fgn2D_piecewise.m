
function U=fgn2D_piecewise(N,H,Variance)
    
    % Generate piecewise homogeneous fractal texture with prescribed
    % local regularity and local variance from a generalized model of
    % Fractional Gaussian Noise combining horizontal and vertical
    % increments of the Fractional Brownian Field
    %
    % from
    % - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for
    % Joint Fractal Feature Estimation and Texture Segmentation,
    % (2019) arxiv:1910.05246
    %
    % Makes use of the implementation of 2D FBM by M. Clausel and B. Vedel,
    % 2008, updated by S. Roux, 2011
    
    l1=1;l2=1;M=N-1;
    if H >= 1
        Zp=[]; fprintf('H must be (strictrly) between 0 and 1'); return
    end
    
    
    %COORDINATES
    X=(-2*2^M:2:2*2^M)/(2^(M+1));
    X(2^M+1)=1/2^M;
    Y=(-2*2^M:2:2*2^M)/(2^(M+1));
    Y(2^M+1)=1/2^M;
    XX=X(ones(1,2*2^M+1),:);
    YY=Y(ones(1,2*2^M+1),:)';
    
    clear X Y
    %rho is the classical pseudonorm associated to the diagonal matrix with eigenvalues  l1 et l2:
    %rho(x,y)=(abs(x)^(2/l1) + abs(y)^(2/l2) )^(1/2)
    rho=sqrt(abs(XX).^(2/l1)+abs(YY).^(2/l2));
    U=rho.^(-l1).*XX;
    V=rho.^(-l2).*YY;
    Geval=sqrt((abs(U)).^(2/l1)+(abs(V)).^(2/l2));
    
    %NOUVELLE DENSITE SPECTRALE PSI
    tau=Geval.*rho;
    
    Hlist = unique(H);
    for iH =1:length(Hlist)
        phi{iH}=tau.^(Hlist(iH)+(l1+l2)/2);
    end
    clear rho tau
    
    %CONSTRUCTION OF THE FOURIER TRANSFORM OF THE FIELD (WITHOUT RENORMALIZATION)
    %W= Fourier transform of the OSSRGF
    %Zr and Zi avoid to include additional symmetries in Fourier
    Zr=randn(2*2^M+1,2*2^M+1);
    Zi=randn(2*2^M+1,2*2^M+1);
    Z = Zr + 1i*Zi;
    for iH =1:length(Hlist)
        W{iH}=fftshift(fft2(Z))./phi{iH};
    end
    clear Z phi
    
    %CONSTRUCTION OF THE FIELD (FOURIER INVERSE + RENORMALIZATION)
    U = zeros(size(H));
    for iH =1:length(Hlist)
        T=real(ifft2(ifftshift(W{iH})));
        Zp{iH}=T-T(2^M+1,2^M+1);
        Zh = 1/2*Zp{iH}(:,1:end-1) - 1/2*Zp{iH}(:,2:end);
        Zv = 1/2*Zp{iH}(1:end-1,:) - 1/2*Zp{iH}(2:end,:);
        Zp{iH} = Zh(1:2^N,1:2^N) + Zv(1:2^N,1:2^N);
        Zp{iH} = Zp{iH}/std2(Zp{iH});
        U(H==Hlist(iH)) =  Zp{iH}(H==Hlist(iH));
    end
    
    
    U = sqrt(Variance).*U;
end
    
    
    
    
    
