% Adjoint operator of opL computing vertical and horizontal gradients of 
% grayscale intensity

function x = opL_adj(y1,y2)
    
    % input   - y1: map of vertical differences
    %         - y2: map of horizontal differences
    %
    % outputs - x: adjoint grayscale image
    %
    % Implementation N. Pustelnik, ENS Lyon
    % June 2019
    
    
    tau = 1;
    y = y1;
    [~,m] = size(y);
    x = -([y(:,1:tau)/2,y(:,1+tau:m-tau)/2- y(:,1:m-2*tau)/2,-y(:,m-2*tau+1:m-tau)/2]);
    y = y2';
    [~,m] = size(y);
    x = x -([y(:,1:tau)/2,y(:,1+tau:m-tau)/2- y(:,1:m-2*tau)/2,-y(:,m-2*tau+1:m-tau)/2])';
    
end

