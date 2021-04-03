% Proximal operator of the least-square data fidelity term encapsulating
% the log-linear behavior of wavelet leaders through scales
%
% from
% - B. Pascal, N. Pustelnik, P. Abry: Strongly Convex Optimization for 
% Joint Fractal Feature Estimation and Texture Segmentation, 
% (2019) arxiv:1910.05246

function [h,v] = prox_hv(ph,ps,use,tau)
    
    
    % Compute the proximal operator of data fidelity term
    %       (h,v) --> 1/2 sum_j || Lj - jh - v ||^2
    % at point (ph,ps)
    %
    % inputs  - ph,ps: current point
    %         - use: computed from usefull.m, contain S0,S1,S2, SLj, SjLj
    %
    % outputs - h,v: proximal point
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    det = (1+tau*use.S0)*(1+tau*use.S2)-(tau*use.S1)^2;
    h = ((1+tau*use.S0)*(ph + tau*use.SjLj)-tau*use.S1*(ps + tau*use.SLj))/det;
    v = ((1+tau*use.S2)*(ps + tau*use.SLj)-tau*use.S1*(ph + tau*use.SjLj))/det;
end