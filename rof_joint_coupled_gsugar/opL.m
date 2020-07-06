% Vertical and horizontal gradients of grayscale intensity


function [xv,xh] = opL(x)
    
    % input   - x: grayscale image
    %
    % outputs - xv: vertical differences
    %         - xh: horizontal differences
    %
    % Implementation N. Pustelnik, ENS Lyon
    % June 2019
    
    tau     = 1;         % finite difference step sise
    [N1,N2] = size(x);   % size of input image x
    
    % Vertical differencex
    xv = [x(:,(1+tau):N2)/2-x(:,1:N2-tau)/2,zeros(N1,tau)];
    
    % Transpose
    x       = x';
    [N1,N2] = size(x);
    
    % Horizontal differences
    xh = [x(:,(1+tau):N2)/2-x(:,1:N2-tau)/2,zeros(N1,tau)]';
    
end