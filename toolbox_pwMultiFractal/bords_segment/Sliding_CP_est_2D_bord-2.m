function [est,valpos] = Sliding_CP_est_2D_bord(leaders,sizePatch,j1,j2)
% Calculate C1 cp for overlapping patches
% REQUIRES leaders computed by DCLx2d_lowmem_bord.m
%
% INPUT :
% - leaders   : calculated by DCLx2d_lowmem_bord.m with J2 >= j2
% - sizePatch : size of (square) window over which CP are computed (in pixels)
%       IMPORTANT: if sizePatch=1, only C1 and c1 are output and correspond to raw log-leaders (no spatial averages)
% - j1,j2     : finest and coarsest scale to perform LF
% OUTPUT :
% est
% - est.C1       : patchwise estimation of C1 for j=j1,...,jmax (Nx x Ny x J)
% - est.C2       : ...
% - est.C3       : ...
% - est.c1       : patchwise estimation of cp ordinary LF over [j1,j2] (Nx x Ny)
% - est.c2
% - est.c3
% - valpos(j)    : indices of valid estimates (no border effects from leaders or spatial average)
%       .valx
%       .valy
%       IMPORTANT: for est.cp, valid estimates are indicated by valpos(j = j2)
%
% HW, IRIT-TLS, 05/2017
% HW, IRIT-TLS, 10/2014

% Filters for calculating Cumulants (i.e., empirical means)
FL=1;  % SWITCH BETWEEN CIRCLE (1) AND SQUARE (2)
PS=sizePatch;
if PS>1;
    switch FL
        case 1 % circle
            x=linspace(-1,1,PS);[X,Y]=meshgrid(x);
            F1=zeros(size(X)); F1(sqrt(X.^2+Y.^2)<=sqrt(2))=1;
        case 2 % square
            F1=ones(PS);
        case 3 % Gaussian
            PS=1.5*PS;
            x=linspace(-1,1,PS); [X,Y]=meshgrid(x); 
            F1=exp(-(X.^2+Y.^2)*1.5);
%             figure(111);clf;imagesc(F1);colorbar
%             s1=sum(F1(:));
%             s2=sum(F1(sqrt(X.^2+Y.^2)<=1/sqrt(2)))/s1
%             keyboard
    end
    F1=F1/sum(F1(:));
else
    F1=1;
end

%% Calculate cumulants
%[a,b]=size(leaders(j1).valueall);
[a,b]=size(leaders(j1).value);
est.h=zeros(a,b);est.c1=zeros(a,b);est.C1=zeros(a,b,j2);
est.C1=zeros(a,b,j2);est.C2=zeros(a,b,j2);est.C3=zeros(a,b,j2);est.c2=zeros(a,b);est.c3=zeros(a,b);

%keyboard

% loop scales: calculate 
for j=1:j2;
    % remove potential zeros and take log
    %LLX=leaders(j).valueall; 
    LLX=leaders(j).value; 
    LLXh=LLX; id0=find(LLX(:)==0);id1=find(LLX(:)~=0); LLX(id0)=min(LLX(id1));
    LLX=log(LLX);
    % replace border leaders (inf) with zeros
    infpos=find(~isfinite(LLX)); LLX(infpos)=0; LLXh(infpos)=0; 
    
    if PS>1 % compute local cumulants
        WW=conv2(ones(size(LLXh)),   F1,'same'); % need to compensate at borders
        est.H(:,:,j)=log2(1./WW.*conv2(LLXh,   F1,'same'));
        est.C1(:,:,j)=1./WW.*conv2(LLX,   F1,'same');
        est.C2(:,:,j)=1./WW.*conv2(LLX.^2,F1,'same') - est.C1(:,:,j).^2;
        est.C3(:,:,j)=1./WW.*conv2(LLX.^3,F1,'same') - 3*est.C2(:,:,j).*est.C1(:,:,j) - est.C1(:,:,j).^3;
        valpos(j).valx=leaders(j).xpos;valpos(j).valy=leaders(j).ypos;
%         valpos(j).valx=leaders(j).xpos(1)+(PS-1):leaders(j).xpos(end)-(PS-1);
%         valpos(j).valy=leaders(j).ypos(1)+(PS-1):leaders(j).ypos(end)-(PS-1);
    else % output leaders
        est.C1(:,:,j)=LLX; 
        est.H(:,:,j)=log2(LLXh);
        valpos(j).valx=leaders(j).xpos;valpos(j).valy=leaders(j).ypos;
    end
    
%    keyboard
end


%% regression
% weights
jj = j1:j2 ;
J  = length(jj); 
wvarjj = ones(1,J);   
S0 = sum(1./wvarjj) ; S1 = sum(jj./wvarjj) ; S2 = sum(jj.^2./wvarjj) ;
wjj = (S0 * jj - S1) ./ wvarjj / (S0*S2-S1*S1) ;

% slopes
ix=leaders(j2).xpos;iy=leaders(j2).ypos;
tmp=est.H(ix,iy,j1:j2); for j=1:j2-j1+1; tmp(:,:,j)=tmp(:,:,j)*wjj(j); end; est.h(ix,iy)=sum(tmp,3);
tmp=est.C1(ix,iy,j1:j2); for j=1:j2-j1+1; tmp(:,:,j)=tmp(:,:,j)*wjj(j); end; est.c1(ix,iy)=sum(tmp,3)/log(2);
if PS>1
    tmp=est.C2(ix,iy,j1:j2); for j=1:j2-j1+1; tmp(:,:,j)=tmp(:,:,j)*wjj(j); end; est.c2(ix,iy)=sum(tmp,3)/log(2);
    tmp=est.C3(ix,iy,j1:j2); for j=1:j2-j1+1; tmp(:,:,j)=tmp(:,:,j)*wjj(j); end; est.c3(ix,iy)=sum(tmp,3)/log(2);
end

%% replace non-valid points by nan
for j=1:j2;
    tmp=est.H(:,:,j);est.H(:,:,j)=nan; est.H(valpos(j).valx,valpos(j).valy,j)=tmp(valpos(j).valx,valpos(j).valy);
    tmp=est.C1(:,:,j);est.C1(:,:,j)=nan; est.C1(valpos(j).valx,valpos(j).valy,j)=tmp(valpos(j).valx,valpos(j).valy);
    tmp=est.C2(:,:,j);est.C2(:,:,j)=nan; est.C2(valpos(j).valx,valpos(j).valy,j)=tmp(valpos(j).valx,valpos(j).valy);
    tmp=est.C3(:,:,j);est.C3(:,:,j)=nan; est.C3(valpos(j).valx,valpos(j).valy,j)=tmp(valpos(j).valx,valpos(j).valy);
    tmp=est.c1;est.c1(:,:)=nan; est.c1(valpos(j2).valx,valpos(j2).valy)=tmp(valpos(j2).valx,valpos(j2).valy);
    tmp=est.h;est.h(:,:)=nan; est.h(valpos(j2).valx,valpos(j2).valy)=tmp(valpos(j2).valx,valpos(j2).valy);
    if PS>1
        tmp=est.c2;est.c2(:,:)=nan; est.c2(valpos(j2).valx,valpos(j2).valy)=tmp(valpos(j2).valx,valpos(j2).valy);
        tmp=est.c3;est.c3(:,:)=nan; est.c3(valpos(j2).valx,valpos(j2).valy)=tmp(valpos(j2).valx,valpos(j2).valy);
    end
end

