function [est,coef] = regu_loc_cp_point_est_dcwt_1d(data, Nwt, j1, j2, gamint, p, WIN,WinLoc)

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
[coef, leaders, nj] = DLPx1dpointloc(data, Nwt, gamint, Inf, csym, j2,WinLoc);
if PLD; [coef2, pleaders, nj] = DLPx1dpointloc(data, Nwt, gamint, p, csym, j2,WinLoc); end

% read coefficients into matrix
WTCOEFS=nan(j2,n);LX=nan(j2,n);pLX=nan(j2,n);LOCX=nan(j2,n);
if BORD==0
    for j=j1:j2
        WTCOEFS(j,coef(j).nbord(1):coef(j).nbord(2))=coef(j).value;
        LX(j,leaders(j).nbord(1):leaders(j).nbord(2))=leaders(j).value;
%         LOCX(j,leaders(j).nbord(1):leaders(j).nbord(2))=coef(j).locsupcoef;
        LOCX(j,leaders(j).nbord(1):leaders(j).nbord(2))=leaders(j).locsupcoef;
%         LOCX(j,leaders(j).nbord(1):leaders(j).nbord(2))=leaders(j).sv.locsupcoef;
        if PLD; pLX(j,pleaders(j).nbord(1):pleaders(j).nbord(2))=pleaders(j).value; end
    end
else
    for j=j1:j2
        WTCOEFS(j,:)=abs(coef(j).value_sbord);
        LX(j,:)=leaders(j).value_sbord;
%         LOCX(j,:)=coef(j).locsupcoef_sb;
        LOCX(j,:)=leaders(j).locsupcoef_sb;
%         LOCX(j,:)=leaders(j).sv.locsupcoef_sb;
        if PLD; pLX(j,:)=pleaders(j).value_sbord; end
    end
end

% perform linear regressions - no window
SJpoint=log2(WTCOEFS); SJpointL=log2(LX); SJpointLoc=log2(LOCX);
yj=SJpoint(:,1:end)'; varyj=ones(size(yj)); [zc,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=SJpointL(:,1:end)'; varyj=ones(size(yj)); [zl,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=SJpointLoc(:,1:end)'; varyj=ones(size(yj)); [zloc,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
est.c=zc(1:end-2^j2); est.l=zl(1:end-2^j2);est.loc=zloc(1:end-2^j2);
est.t=[1:length(est.c)];
if PLD; SJpointLp=log2(pLX); yj=SJpointLp(:,1:end)'; varyj=ones(size(yj)); [zp,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2); est.p=zp(1:end-2^j2); end


% WINDOW
wo=ceil(WIN/2);
W_LOCX_M=nan(size(LX)); W_LX_M=nan(size(LX)); W_DX_M=nan(size(WTCOEFS)); W_PX_M=nan(size(pLX));
Wconv=ones(1,WIN)/WIN;
for kj=1:j2
    W_LOCX_M(kj,:)=minmaxfilt1(LOCX(kj,:),WIN,'max','same');
    W_LX_M( kj, wo:end-WIN+wo ) =conv(squeeze(LX( kj, : )),Wconv,'valid');
    W_DX_M( kj, wo:end-WIN+wo )=conv(squeeze(WTCOEFS( kj, : )),Wconv,'valid');
    if PLD; W_PX_M( kj, wo:end-WIN+wo ) =conv(squeeze(pLX( kj, : )),Wconv,'valid'); end
    id=find(isnan(W_LX_M( kj, n/2+1:end) )); W_LOCX_M(kj,n/2+id(1):end)=nan;
end

% cumulants
wo=ceil(WIN/2);
W_LX_C1=nan(size(LX)); W_LX_C2=nan(size(LX));
for kj=1:j2
    W_LX_C1( kj, wo:end-WIN+wo ) =conv(log(abs(squeeze(LX( kj, : )))),Wconv,'valid');
    W_LX_C2( kj, wo:end-WIN+wo ) =conv(log(abs(squeeze(LX( kj, : )))).^2,Wconv,'valid') - W_LX_C1( kj, wo:end-WIN+wo ).^2;
end
W_LX_C1=W_LX_C1/log(2);
W_LX_C2=W_LX_C2/log(2);

% perform linear regressions - windowed
SJpoint=log2(W_DX_M); SJpointL=log2(W_LX_M);SJpointLoc=log2(W_LOCX_M);SJ_C1=(W_LX_C1);SJ_C2=(W_LX_C2);
yj=SJpoint(:,1:end)'; varyj=ones(size(yj)); [zc,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=SJpointL(:,1:end)'; varyj=ones(size(yj)); [zl,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=SJpointLoc(:,1:end)'; varyj=ones(size(yj)); [zloc,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=SJ_C1(:,1:end)'; varyj=ones(size(yj)); [c1,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
yj=SJ_C2(:,1:end)'; varyj=ones(size(yj)); [c2,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);
est.cw=zc(1:end-2^j2); est.lw=zl(1:end-2^j2); est.locw=zloc(1:end-2^j2); est.c1=c1(1:end-2^j2);  est.c2=c2(1:end-2^j2);
if PLD; SJpointLp=log2(W_PX_M);yj=SJpointLp(:,1:end)'; varyj=ones(size(yj)); [zp,Vzeta,Q,aest]=MFA_BS_regrmat(yj,varyj,nj,wtype, j1,j2);est.pw=zp(1:end-2^j2);end


% keyboard