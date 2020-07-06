% Power method to compute the norm of weighted discrete differences 
% (wv,wh) -> [opL(wv), alpha opL(wh)] (signal)


function ND = normD_1D(alpha,size)
    
    % inputs  - alpha: weight (regularization parameter)
    %         - size: size of the signals wv and wh
    %
    % ouputs  - ND: operator norm obtained as the quare root of the largest
    %         eigen value of opL_adj_1D o opL_1D
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    xv = randn(size);
    xh = randn(size);
    
    
    wv = xv;
    wh = xh;
    K = 200;
    N = zeros(1,K);
    for i = 1:K
        yv = opL_1D(wv);
        yh = opL_1D(wh); yh = alpha*yh;
        zv = opL_adj_1D(yv);
        zh = opL_adj_1D(yh); zh = alpha*zh;
        wv = zv; wh = zh;
        N(i) = norm([zv,zh],'fro');
        wv = wv/N(i); wh = wh/N(i);
    end
    ND = sqrt(N(end));
    
end