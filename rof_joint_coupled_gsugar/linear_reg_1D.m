% Perform a linear regression on the multiscale quantity Lj (each Lj{j}
% being a M pooints signal) to fit the behavior Lj = v + jh, 
% i.e., minimize least squares
% (vreg,hreg) = argmin_{v,h} sum_{j in JJ} || Lj - v - jh ||^2   
%
%
% from 
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246

function [vreg, hreg] = linear_reg_1D(Lj, JJ)

    % inputs  - Lj: multiscale quantity on which the linear regression is
    %         performed
    %         - JJ range of considered scales
    %
    % outputs - vreg: sequence of local intercept
    %         - hreg: sequence of local slopes
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    
    % Number of pixels
    M = numel(Lj{JJ(1)});
    
    %Common quantities for regression
    S0=sum(JJ.^0);
    S1=sum(JJ);
    S2=sum(JJ.^2);
    det = S2*S0 - S1*S1;
    
    % Take the log2 of the leaders
    logLj = zeros(JJ(end),M);

    for jj=JJ
        logLj(jj,:) = Lj{jj};
    end
    

    %Linear regression to find sigma and h from leaders
    SLj = sum(logLj(JJ,:),1);
    SjLj = JJ*logLj(JJ,:);
    hreg = (S0*SjLj - S1*SLj)/det;
    vreg = (-S1*SjLj + S2*SLj)/det;
    
end