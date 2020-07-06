% Power method to compute the norm of weighted discrete gradient 
% (wv,wh) -> [opL(wv), alpha opL(wh)] (image)

function ND = normD_2D(alpha,size)
    
    % inputs  - alpha: weight (regularization parameter)
    %         - size: size of the maps wv and wh
    %
    % ouputs  - ND: operator norm obtained as the quare root of the largest
    %         eigen value of opL_adj o opL
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
        [yv_h,yv_v] = opL(wv);
        [yh_h,yh_v] = opL(wh); 
        yh_h = alpha*yh_h;
        yh_v = alpha*yh_v;
        zv = opL_adj(yv_h,yv_v);
        zh = opL_adj(yh_h,yh_v); 
        zh = alpha*zh;
        wv = zv; wh = zh;
        N(i) = norm([zv,zh],'fro');
        wv = wv/N(i); wh = wh/N(i);
    end
    ND = sqrt(N(end));
    
end