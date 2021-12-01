% Iterated post-processing thresholding providing a segmentation from a
% Total Variation denoised estimate. (image or signal)
%
% from X. Cai and G. Steidl. Multiclass segmentation by iterated ROF thresholding. In Int.Workshop
% on Energy Minimization Methods in Comp. Vis. and Pat. Rec., pages 237?250. Springer, 2013

function [segh,resth]= trof_NaN(h,K)
    
    % inputs  - h: denoised estimate to be segmented
    %         - K: number of regions
    % 
    % outputs - segh: map of label
    %         - resth: piecewise constant estimate
    %
    % Implementation B. Pascal, ENS Lyon
    % June 2019
    
    % THRESHOLD RUDIN OSHER FATEMI
    % From Cai and Chan 
    
    
    % INITIALISATION
    iter = 12; %number of iterations
    max(h(~isnan(h)))
    tau = linspace(min(h(~isnan(h))),max(h(~isnan(h))),K);
    m0 = mean(h(tau(1) >= h));
    ind = cell(1,K-1);
    m = zeros(1,K-1);
    
    % TROF ALGORITHM
    for it = 1:iter
        for i = 1:K-1
            ind{i} = find((h > tau(i))&(tau(i+1) >= h));
            m(i) = mean(h(ind{i}));
            if i ~=1
                tau(i) = 1/2*(m(i-1) + m(i));
            else
                tau(i) = 1/2*(m0 + m(i));
            end
            m0 = mean(h(tau(1) >= h));
        end
    end
    segh = ones(size(h));
    resth = m0*ones(size(h));
    for i = 1:K-1
        segh(h > tau(i)) = i+1;
        resth(h > tau(i)) = m(i);
    end
    
    
    
end