% Chambolle-Pock primal-dual algorithm with strong convexity acceleration
% to perform Total Variation denoising (image)
%
% from A. Chambolle, T. Pock: A first-order primal-dual algorithm for convex 
% problems with applications to imaging. J. Math. Imag. Vis. 40(1), 
% 120-145 (2011)


function [x, crit, gap, t]=PA_PDtv(I,prox)
    
    % inputs  - I.img_n: observed image to be denoised
    %         - prox.lambda: regularization parameter
    %
    % ouputs  - x: denoised image
    %         - t: time
    %         - crit: evolution of objective function
    %         - gap: evolution of duality gap
    %
    % Implementation B. Pascal, ENS Lyon
    % May 2020
    
    img_n  = I.img_n;
    
    %% Fixing Proximal Parameters
    
    gamma = 0.99;
    mu_g=1;
    normD = sqrt(2);
    tau = gamma/normD;
    sig = gamma/normD;
    lambda = prox.lambda;
    iter = 1e5;
    eps = 1e-4;
    
    
    %% Initializing variables
    
    
    x= sparse(zeros(size(img_n)));
    [tmph,tmpv] = opL(img_n);
    yh=sparse(zeros(size(tmph)));
    yv=sparse(zeros(size(tmpv)));
    byh=sparse(zeros(size(tmph)));
    byv=sparse(zeros(size(tmpv)));
    
    
    %% Criterion of convergence
    crit=zeros(1,iter);
    gap=zeros(1,iter);
    t=zeros(1,iter);
    it = 0;
    gapc = eps+1;
    
    %% Algorithm
    while (gapc > eps)&&(it<iter)
        
        it = it + 1;
        tic
        %Save the dual variables
        yhs = yh;
        yvs = yv;
        
        
        %Update of primal variable
        x = x - tau*opL_adj(byh,byv);
        x = prox_g(x, img_n,tau);
        
        
        %Update of dual variable
        [th, tv] = opL(x);
        tilyh = yhs + sig * th;
        tilyv = yvs + sig * tv;
        [temph, tempv] = prox_L12(tilyh/sig, tilyv/sig, lambda/sig);
        
        yh = tilyh - sig*temph;
        yv = tilyv - sig*tempv;
        
        %Update of the descent steps
        theta = (1+2*mu_g*tau)^(-1/2);
        tau = theta*tau;
        sig=sig/theta;
        
        %Update dual auxiliary variable
        byh = yh + theta*(yh - yhs);
        byv = yv + theta*(yv - yvs);
        
        
        
        t(it) = toc;
        
        % Compute primal criterion
        [ph,pv] = opL(x);
        crit(it) = 1/2*norm(x-img_n,'fro')^2 + lambda*sum(sum((ph.^2+pv.^2).^(1/2)));
        
        % Compute dual criterion
        p = opL_adj(yh,yv);
        [p1, p2] = prox_L12(yh,yv,lambda);
        dual = 1/2*norm(-p ,'fro')^2 - sum(p(:).*img_n(:)) + norm([p1,p2],'fro');
        
        % Compute duality gap
        gap(it) = crit(it) + dual;
        gapc = 2*gap(it)/(abs(crit(it)) + abs(dual));
        
        
        
    end
    
    
    t = t(1:it);
    crit = crit(1:it);
    gap = gap(1:it);
    
    t = cumsum(t);
end
