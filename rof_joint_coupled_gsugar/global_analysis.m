% Global region per region estimation of the local regularity, from linear
% regression over averaged wavelet coefficients or leader wavelet
% coefficients.

function G = global_analysis(X, L, seg)
    
    % inputs   - X: textured image or signal to be segmented
    %          - L: wavelet coefficients and leaders computed by
    %          multiscale_analysis(X)
    %          - seg: segmentation into K regions or segments
    % 
    % outputs: - G.Y: estimated regularities from globaly averaged coefficients
    %          - G.L: estimated regularities from globaly averaged leaders
    
    % Number of regions
    K       = length(unique(seg));
    
    % Range of considered scales
    JJ      = L.JJ;
    tJJ     = zeros(1,JJ(end));
    tJJ(JJ) = JJ;
    
    % From coefs to structure function
    SY     = zeros(K,JJ(end));
    Sy = SY;
    for jj = JJ
        for k = 1:K
            SY(k,jj) = mean(L.coefs{jj}(seg(:) == k).^2);
            Sy(k,jj) = log2(SY(k,jj)); 
        end
    end
    
    % From leaders to structure function
    SL     = zeros(K,JJ(end));
    Sell = SL;
    for jj = JJ
        for k = 1:K
            SL(k,jj)  = mean(L.leaders{jj}(seg(:) == k).^2);
            Sell(k,jj) = log2(SL(k,jj)); 
        end
    end
   
    
    % Common quantities for regression
    S0     = sum(JJ.^0);
    S1     = sum(JJ);
    S2     = sum(JJ.^2);
    det    = S2*S0 - S1*S1;
    
    % Linear regression to find sigma and h from coefs
    Syj  = sum(Sy,2);
    Sjyj = Sy*tJJ';
    G.Y.h = (S0*Sjyj  - S1*Syj)/(2*det);
    G.Y.v = (-S1*Sjyj + S2*Syj)/(2*det);
    
    % Linear regression to find sigma and h from leaders
    Sellj  = sum(Sell,2);
    Sjellj = Sell*tJJ';
    G.L.h = (S0*Sjellj  - S1*Sellj)/(2*det);
    G.L.v = (-S1*Sjellj + S2*Sellj)/(2*det);
    
    Var = zeros(1,K);
    for k = 1:K
       Var(k) = var(X(seg(:) == k)); 
    end
    G.Var = Var;
    
    disp(' ')
    disp('Posterior global estimates')
    for k = 1:K
       disp(['Texture ',num2str(k),': H=',num2str(G.Y.h(k),3), '(coefs), H=',num2str(G.L.h(k),3), '(leaders),  Var=',num2str(Var(k),3)])
    end
    
end