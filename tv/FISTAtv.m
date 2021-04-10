% Fast Iterative Soft Thresholding Algorithm for the minimization of the
% dual Rudin-Osher-Fatemi functional in order to perform Total Variation
% denoising (image)
%
% from
% - A.Chambolle, C. Dossal: On the convergence of the iterates of ``FISTA",
%  J. Optim. Theory Appl. 166(3), 25 (2015)


function [x, crit, gap, t]=FISTAtv(I,prox)
    
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
    normD  = sqrt(2);
    gamman = gamma/(normD)^2;
    lambda = prox.lambda;
    iter = 1e5;
    eps = 1e-4;
    a=4; % inertia parameter of FISTA
    
    
    %% Initializing variables
    yh = zeros(size(img_n));
    yv = zeros(size(img_n));
    byh = yh;
    byv = yv;
    x = img_n - opL_adj(yh,yv);
    fista = 1;
    
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
        yhs = yh; byhs = byh;
        yvs = yv; byvs = byv;
        
        %Update of dual variable
        [th, tv] = opL(x);
        tilyh = yhs + gamman * th;
        tilyv = yvs + gamman * tv;
        [temph, tempv] = prox_L12(tilyh/gamman, tilyv/gamman, lambda/gamman);
        byh = tilyh - gamman*temph;
        byv = tilyv - gamman*tempv;
        
        %Update of inertia FISTA parameter
        fistas = fista;
        fista = (it+a)/a;
        
        %Update
        yh = byh + (fistas-1)/fista*(byh-byhs);
        yv = byv + (fistas-1)/fista*(byv-byvs);
        
        
        %Update primal variable
        x = x - opL_adj(yh-yhs,yv-yvs);
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
