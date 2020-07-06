% Adjoint operator of opL_1D computing discrete gradient


function x = opL_adj_1D(xt)
    
    % input   - xt: sequence of discrete differences
    %
    % outputs - x: adjoint signal
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    x = zeros(size(xt));
    x(1) = -xt(1)/2;
    x(2:end-1) = xt(1:end-2)/2-  xt(2:end-1)/2;
    x(end) = xt(end-1)/2;
end