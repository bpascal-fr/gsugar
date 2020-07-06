% Perform a linear regression on the multiscale quantity Lj (each Lj{j}
% being a N1 x N2 pixels map) to fit the behavior Lj = v + jh, 
% i.e., minimize least squares
% (vreg,hreg) = argmin_{v,h} sum_{j in JJ} || Lj - v - jh ||^2   

function [vreg, hreg] = linear_reg(Lj, JJ)
    
    % inputs  - Lj: multiscale quantity on which the linear regression is
    %         performed
    %         - JJ range of considered scales
    %
    % outputs - vreg: map of local intercept
    %         - hreg: map of local slopes
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020

    
    % Number of pixels
    M = numel(Lj{JJ(1)});
    [N1, N2] = size(Lj{JJ(1)});
    
    %Common quantities for regression
    S0=sum(JJ.^0);
    S1=sum(JJ);
    S2=sum(JJ.^2);
    det = S2*S0 - S1*S1;
    
    % Take the log2 of the leaders
    logLj = zeros(JJ(end),M);

    for jj=JJ
        logLj(jj,:) = reshape(Lj{jj},1,M);
    end
    

    %Linear regression to find sigma and h from leaders
    SLj = sum(logLj(JJ,:),1);
    SjLj = JJ*logLj(JJ,:);
    hreg = (S0*SjLj - S1*SLj)/det; hreg = reshape(hreg,N1, N2);
    vreg = (-S1*SjLj + S2*SLj)/det; vreg = reshape(vreg,N1, N2);
    
end