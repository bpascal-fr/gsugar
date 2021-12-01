% Perform the multiscale analysis, computing the leader wavelet
% coefficients from Daubechies wavelets with Nwt vanishing moments, compute
% the linear regression estimates of local regularity h and local power v.
%
% Require the use of provided toolbox_pwMultiFractal developped by H. Wendt
% (see https://www.irit.fr/~Herwig.Wendt/software.html)
%
%
% from 
% - S. Jaffard. Wavelet techniques in multifractal analysis. Fractal Geometry
% and Applications: A Jubilee of Benoit Mandelbrot, M. Lapidus and M.
% van Frankenhuijsen Eds., Proceedings of Symposia in Pure Mathematics,
% 72(2):91{152, 2004.
% and
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246

function [L,C] = multiscale_analysis_light(X, JJ, Nwt, gamint)
    
    % inputs  - X: textured image to be analyzed
    %         - JJ: range of scales considered (default 1:3)
    %         - Nwt : number of vanishing moments of wavelet (default 2)
    %         - gamint: fractional integration parameter (default 1)
    %
    % outputs L: quantities computed from wavelet leaders 
    %         - L.leaders, log2 leaders coefficients of X
    %         - L.coefs, absolute value of maximal wavelet coefficients of X
    %         - L.h_LR, linear regression estimate of local regularity
    %         - L.v_LR, linear regression estimate of local power
    %         - L.JJ, range of scales considered
    %         C: quantities computed from wavelet coefficients 
    %         - C.coefs, absolute value of maximal wavelet coefficients of X
    %         - C.h_LR, linear regression estimate of local regularity
    %         - C.v_LR, linear regression estimate of local power
    %         - C.JJ, range of scales considered
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    
    if nargin == 1
        JJ = 1:3;
        Nwt = 2;
        gamint = 1;
    elseif nargin == 2
        Nwt = 2;
        gamint = 1;
    elseif nargin == 3
        gamint = 1;
    end
    
    % Store range of scales
    L.JJ = JJ;
    C.JJ = JJ;
    J2 = JJ(end);
    
    % COMPUTATION OF THE CUMULANTS AND MULTIFRACTALS PARAMETERS
    
    % Computation
    [coefs, leaders, ~] = DCLx2d_lowmem_bord(X, Nwt, gamint,0, J2, 0); %leaders at each point (no decimation) with Lambda 3 (no modification for the moment)
    [N1,N2] = size(X);
    M =  numel(leaders(1).value(:));
    
    
    % Extraction of maximum wavelet coefficients
    Yj = zeros(3,M,J2);
    Cj = zeros(J2,M);
    for jj=1:J2
        for m =1:3
            Yj(m,:,jj) = reshape(abs(coefs(jj).value(:,:,m)),1,M);
        end
        Cj(jj,:) = reshape(max(Yj(:,:,jj)),1,M);
        L.coefs{jj}= reshape(max(Yj(:,:,jj)),N1,N2);
        C.coefs{jj}=reshape(max(Yj(:,:,jj)),N1,N2);
    end
    
    
    % Extraction of leaders
    Lj = zeros(J2,M);
    for jj=1:J2
        Lj(jj,:) = reshape(log2(leaders(jj).value),1,M);
        L.leaders{jj}=leaders(jj).value;
    end
    
    
    % Common quantities for regression
    S0=sum(JJ.^0);
    S1=sum(JJ);
    S2=sum(JJ.^2);
    det = S2*S0 - S1*S1;
    
    
    % Linear regression to find sigma and h from leaders
    SLj = sum(Lj(JJ,:),1);
    SjLj = (JJ)*Lj(JJ,:);
    h_LR = (S0*SjLj - S1*SLj)/det;
    v_LR = (-S1*SjLj + S2*SLj)/det;
    L.h_LR = reshape(h_LR,N1,N2);
    L.v_LR = reshape(v_LR,N1,N2);
    
    % Linear regression to find sigma and h from coefficients
    SCj = sum(Cj(JJ,:),1); 
    SjCj = (JJ)*Cj(JJ,:);
    hc_LR = (S0*SjCj - S1*SCj)/det;
    vc_LR = (-S1*SjCj + S2*SCj)/det;
    C.h_LR = reshape(hc_LR,N1,N2);
    C.v_LR = reshape(vc_LR,N1,N2);
    
end