% Compute discrete differences of a signal
    
function xt = opL_1D(x)
    
    
    % input   - x: signal
    %
    % outputs - xt: discrete differences
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    
    xt = zeros(size(x));
    xt(1:end-1) = x(2:end)/2 - x(1:end-1)/2;
    
end
