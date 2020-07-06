% Perform the multiscale analysis, computing the leader wavelet
% coefficients from Daubechies wavelets with Nwt vanishing moments, compute
% the linear regression estimates of local regularity h and local power v,
% estimate the covariance structure of log-leaders. (signal)
%
% Require the use of provided toolbox_pwMultiFractal developped by H. Wendt
% (see https://www.irit.fr/~Herwig.Wendt/software.html)

function L = multiscale_analysis_1D(X, JJ, Nwt, gamint)
    
    % inputs  - X: textured image to be analyzed
    %         - JJ: range of scales considered (default 1:3)
    %         - Nwt : number of vanishing moments of wavelet (default 2)
    %         - gamint: fractional integration parameter (default 1)
    %
    % outputs - L.leaders, log2 leaders coefficients of X
    %         - L.coefs, absolute value of maximal wavelet coefficients of X
    %         - L.S, estimated covariance matrix of leaders
    %         - L.h_LR, linear regression estiamte of local regularity
    %         - L.v_LR, linear regression estimate of local power
    %         - L.JJ,range of scales considered
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
    J2 = JJ(end);
    
    % COMPUTATION OF THE CUMULANTS AND MULTIFRACTALS PARAMETERS
    
    % Computation
    [coefs, leaders, ~] = DLPx1dloc(X, Nwt, gamint, Inf, 0);
    M = numel(leaders(1).value_sbord(:));
    
    % Extraction of maximum wavelet coefficients
    Yj = zeros(3,M,J2);
    for jj=1:J2
        for m =1:3
            Yj(m,:,jj) = abs(coefs(jj).value_sbord(:,m));
        end
        L.coefs{jj}=max(Yj(:,:,jj));
    end
    
    % Extraction of leaders
    Lj = zeros(J2,M);
    for jj=1:J2
        Lj(jj,:) = log2(leaders(jj).value_sbord);
        L.leaders{jj}=leaders(jj).value_sbord;
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
    L.h_LR = h_LR;
    L.v_LR = v_LR;
    
    
    
    
    % Estimate the covariance structure of leaders
    S = cell(length(JJ),length(JJ)); % covariance matrix
    D = zeros(1, length(JJ));        % variance of log2-leaders
    for ii = JJ
        for jj = JJ
            % Estimate correlations 
            S{ii,jj} = xcorr(log2(L.leaders{ii}),log2(L.leaders{jj}))/M - mean(log2(L.leaders{ii}(:)))*mean(log2(L.leaders{jj}(:)));
            % Truncate it
            s_cov = 2^(ii-1)+2^(jj-1);
            S{ii,jj} = S{ii,jj}(M-s_cov:M+s_cov);
        end
        D(ii) = var(log2(L.leaders{ii}(:)));
    end
    L.S = S;
    L.D = D;
end