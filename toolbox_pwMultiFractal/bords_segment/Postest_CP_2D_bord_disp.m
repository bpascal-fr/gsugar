function [est,c1,c2] = Postest_CP_2D_bord_disp(leaders,MASK,j1,j2,who,gamint)
%% function [est] = Postest_CP_2D_bord(leaders,MASK,j1,j2)
% Posteriori estimation of log-cumulants in zones defined by constant
% values of MASK.
%
% INPUT :
% - leaders   : calculated by DCLx2d_lowmem_bord.m with J2 >= j2
% - MASK      : matrix of size size(leaders(1).value) with as entries a unique value
%               for each homogeneous zone
% - j1,j2     : finest and coarsest scale to perform LF
% - who       : string of characters, which mask is used to perform Postest
% OUTPUT :
% est
% - est.C1       : zone-wise estimation of C1 for j=j1,...,jmax (Nzones x J)
% - est.C2       : ...
% - est.C3       : ...
% - est.c1       : zone-wise estimation of cp ordinary LF over [j1,j2] (Nzones x 1)
% - est.c2       : ...
% - est.c3       : ...
% - est.MaskID   : the unique values found in MASK, in the order used for (Nzones x 1)
%                  estimates
%
% HW, IRIT-TLS, 05/2017

MID=unique(MASK(:)); % values for different regions in mask
ML=length(MID);

[a,b]=size(leaders(j1).value);
est.h=zeros(ML,1);est.c1=zeros(ML,1);est.C1=zeros(ML,j2);
est.C1=zeros(ML,j2);est.C2=zeros(ML,j2);est.C3=zeros(ML,j2);est.c2=zeros(ML,1);est.c3=zeros(ML,1);
est.MaskID=MID;

%% Calculate cumulants
% loop scales
for j=1:j2;
    % remove potential zeros and take log
    LLX=leaders(j).value; 
    LLXh=LLX; id0=find(LLX(:)==0);id1=find(LLX(:)~=0); LLX(id0)=min(LLX(id1));
    LLX=log(LLX);
    % replace border leaders (inf) with zeros
    infpos=find(~isfinite(LLX)); LLX(infpos)=0; LLXh(infpos)=0; 

    % loop mask zones
    for i=1:length(MID)
        est.H(i,j)=log2(mean(LLXh(MASK==MID(i))));
        est.C1(i,j)=mean(LLX(MASK==MID(i)));
        est.C2(i,j)=mean(LLX(MASK==MID(i)).^2) - est.C1(i,j).^2;
        est.C3(i,j)=mean(LLX(MASK==MID(i)).^3) - 3*est.C2(i,j).*est.C1(i,j) - est.C1(i,j).^3;
    end
end

%% regression
% weights
jj = j1:j2 ; J  = length(jj);  wvarjj = ones(1,J);   
S0 = sum(1./wvarjj) ; S1 = sum(jj./wvarjj) ; S2 = sum(jj.^2./wvarjj) ;
wjj = (S0 * jj - S1) ./ wvarjj / (S0*S2-S1*S1) ;

% slopes
tmp=est.H(:,j1:j2);  for j=1:j2-j1+1; tmp(:,j)=tmp(:,j)*wjj(j); end; est.h=sum(tmp,2);
tmp=est.C1(:,j1:j2); for j=1:j2-j1+1; tmp(:,j)=tmp(:,j)*wjj(j); end; est.c1=sum(tmp,2)/log(2);
tmp=est.C2(:,j1:j2); for j=1:j2-j1+1; tmp(:,j)=tmp(:,j)*wjj(j); end; est.c2=sum(tmp,2)/log(2);
tmp=est.C3(:,j1:j2); for j=1:j2-j1+1; tmp(:,j)=tmp(:,j)*wjj(j); end; est.c3=sum(tmp,2)/log(2);

c1 = est.c1-gamint;
c2 = est.c2;
disp(['Using ', who , ' classification, the c1 are ',num2str(est.c1'-1), ' and the c2 are ', num2str(est.c2')])