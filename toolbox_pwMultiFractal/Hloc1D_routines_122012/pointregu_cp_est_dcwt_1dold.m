function [est,structF,coef] = pointregu_cp_est_dcwt_1d(data, Nwt, j1, j2, gamint, p, WIN)

BORD=0; % 1: keep polluted borders
PLD=0;  % 1: compute also p-Leaders

Norm=1; wtype=0;

if nargin<5; gamint=0; end;
if nargin<6; p=2; end;
if nargin<7; WIN=1; end;

%-- Initialize the wavelet filters
n = length(data) ;                   % data length

%-- compute dcwt with symmetric wavelet
csym=1;
[coef, leaders, nj] = DLPx1dloc(data, Nwt, gamint, Inf, csym, j2);
if PLD; [coef2, pleaders, nj] = DLPx1dloc(data, Nwt, gamint, p, csym, j2); end


% read coefficients into matrix
WTCOEFS=nan(j2,n);LX=nan(j2,n);pLX=nan(j2,n);
if BORD==0
    for j=j1:j2
        WTCOEFS(j,coef(j).nbord(1):coef(j).nbord(2))=coef(j).value;
        LX(j,leaders(j).nbord(1):leaders(j).nbord(2))=leaders(j).value;
        if PLD; pLX(j,pleaders(j).nbord(1):pleaders(j).nbord(2))=pleaders(j).value; end
    end
else
    for j=j1:j2
        WTCOEFS(j,:)=abs(coef(j).value_sbord);
        LX(j,:)=leaders(j).value_sbord;
        if PLD; pLX(j,:)=pleaders(j).value_sbord; end
    end
end

% perform linear regressions - no window
SJpoint=log2(WTCOEFS); SJpointL=log2(LX);
yj=SJpoint(:,1:end)'; varyj=ones(size(yj)); [zc,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=SJpointL(:,1:end)'; varyj=ones(size(yj)); [zl,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
zc(end-2^j2+1:end)=nan; zl(end-2^j2+1:end)=nan;
est.c=zc; est.l=zl;
% est.c=zc(1:end-2^j2); est.l=zl(1:end-2^j2);
est.t=[1:length(est.c)];
structF.cSJwin=SJpoint;
structF.lSJwin=SJpointL;

if PLD; structF.pSJwin=SJpointL; SJpointLp=log2(pLX); yj=SJpointLp(:,1:end)'; varyj=ones(size(yj)); [zp,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2); est.p=zp(1:end-2^j2); end


% WINDOW
if WIN>1; wo=ceil(WIN/2); WIN=2*wo; end
W_LX_M=nan(size(LX)); W_DX_M=nan(size(WTCOEFS)); W_PX_M=nan(size(pLX));
cC1=nan(size(LX));cC2=nan(size(LX));cC3=nan(size(LX));lC1=nan(size(LX));lC2=nan(size(LX));lC3=nan(size(LX));pC1=nan(size(LX));pC2=nan(size(LX));pC3=nan(size(LX));
Wconv=ones(1,WIN)/WIN;
for kj=j1:j2
    W_LX_M( kj, wo:end-WIN+wo ) =conv(squeeze(LX( kj, : )),Wconv,'valid');
    W_DX_M( kj, wo:end-WIN+wo )=conv(squeeze(WTCOEFS( kj, : )),Wconv,'valid');
    if PLD; W_PX_M( kj, wo:end-WIN+wo ) =conv(squeeze(pLX( kj, : )),Wconv,'valid'); end
    % c1, c2
    lLX=log(LX);
    lC1( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )),Wconv,'valid');
    lC2( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )).^2,Wconv,'valid') - lC1( kj, wo:end-WIN+wo ).^2;
    lC3( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )).^3,Wconv,'valid') -3*lC2( kj, wo:end-WIN+wo ).*lC1( kj, wo:end-WIN+wo ) - lC1( kj, wo:end-WIN+wo ).^3;
    W_DX_M( kj, wo:end-WIN+wo )=conv(squeeze(WTCOEFS( kj, : )),Wconv,'valid');
    lLX=log(WTCOEFS);
    cC1( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )),Wconv,'valid');
    cC2( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )).^2,Wconv,'valid') - cC1( kj, wo:end-WIN+wo ).^2;
    cC3( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )).^3,Wconv,'valid') -3*cC2( kj, wo:end-WIN+wo ).*cC1( kj, wo:end-WIN+wo ) - cC1( kj, wo:end-WIN+wo ).^3;
    if PLD; 
        lLX=log(pLX);
        pC1( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )),Wconv,'valid');
        pC2( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )).^2,Wconv,'valid') - pC1( kj, wo:end-WIN+wo ).^2;
        pC3( kj, wo:end-WIN+wo ) =conv(squeeze(lLX( kj, : )).^3,Wconv,'valid') -3*pC2( kj, wo:end-WIN+wo ).*pC1( kj, wo:end-WIN+wo ) - pC1( kj, wo:end-WIN+wo ).^3;
    end
end



% cp
yj=cC1(:,1:end)'; varyj=ones(size(yj)); [z1,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=cC2(:,1:end)'; varyj=ones(size(yj)); [z2,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=cC3(:,1:end)'; varyj=ones(size(yj)); [z3,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
z1(end-2^j2+1:end)=nan; z2(end-2^j2+1:end)=nan; z3(end-2^j2+1:end)=nan;
est.cc1w=z1/log(2); est.cc2w=z2/log(2); est.cc3w=z3/log(2);
% est.cc1w=z1(1:end-2^j2)/log(2); est.cc2w=z2(1:end-2^j2)/log(2); est.cc3w=z3(1:end-2^j2)/log(2);
structF.cC1win=cC1;structF.cC2win=cC2; structF.cC3win=cC3;
yj=lC1(:,1:end)'; varyj=ones(size(yj)); [z1,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=lC2(:,1:end)'; varyj=ones(size(yj)); [z2,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=lC3(:,1:end)'; varyj=ones(size(yj)); [z3,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
z1(end-2^j2+1:end)=nan; z2(end-2^j2+1:end)=nan; z3(end-2^j2+1:end)=nan;
est.lc1w=z1/log(2); est.lc2w=z2/log(2); est.lc3w=z3/log(2);
% est.lc1w=z1(1:end-2^j2)/log(2); est.lc2w=z2(1:end-2^j2)/log(2); est.lc3w=z3(1:end-2^j2)/log(2);
structF.lC1win=lC1;structF.lC2win=lC2;structF.lC3win=lC3;
if PLD
    yj=pC1(:,1:end)'; varyj=ones(size(yj)); [z1,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
    yj=pC2(:,1:end)'; varyj=ones(size(yj)); [z2,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
    yj=pC3(:,1:end)'; varyj=ones(size(yj)); [z3,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
    z1(end-2^j2+1:end)=nan; z2(end-2^j2+1:end)=nan; z3(end-2^j2+1:end)=nan;
    est.pc1w=z1/log(2); est.pc2w=z2/log(2); est.pc3w=z3/log(2);
%     est.pc1w=z1(1:end-2^j2)/log(2); est.pc2w=z2(1:end-2^j2)/log(2); est.pc3w=z3(1:end-2^j2)/log(2);
    structF.pC1win=pC1;structF.pC2win=pC2;structF.pC3win=pC3;
end

