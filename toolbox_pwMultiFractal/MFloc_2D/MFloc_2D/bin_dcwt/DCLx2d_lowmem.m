function [leaders, nj] = DCLx2d_lowmem(data, Nwt, gamint,symm, J2)

%
%  [coef, leaders, nj] = DCLx2d(data, Nwt [,gamint,sym])
%
% compute 2D Dyadic Scale Continuous Time wavelet and leader coefficients
%
% gamint -- fractional integration of order gamint is applied
% sym=0  -- use Daubechies Wavelet of order Nwt (default).
% sym=1  -- use Daubechies Symmetrized Wavelet of order Nwt.
% 
%% INPUT
% data      - image (arbitrarily size)
% Nwt       - # vanishing moments
% gamint    - fractional integration parameter  [default = 0]
% symm      - 0: use Daubechies                 [default = 0]
%           - 1: symmetrize Daubechies
% J2        - coarsest scale to compute         [default: largest available scale]
%
%
%%%%%%%%%%
% HW, IRIT-TLS, 20/2014
%   following
% SR, ens-Lyon, 11/2013
%%%%%%%%%%

Norm=1;


% NHlam=1; % 1 LAMBDA
NHlam=3; % 3 LAMBDA

%-- paramater for  fractional integration
if nargin<3; gamint=0; end;
if isempty(gamint); gamint=0; end;

if nargin<4; symm=0; end;
if nargin<5; J2=inf; end;

%-- Initialize the wavelet filters
n = length(data) ;         % data length
Nwt=abs(Nwt); h = rlistcoefdaub(Nwt) ; nu = 0 ;
if symm == 0  % Daubechies Wavelet
    [v,nv] = tildeurflippeur(h,nu);  nv = nv + 1;
    nl=length(h);
    % parameter for the centering of the wavelet
    x0=2;
    x0Appro=2*Nwt;
else % Daubechies Symmetrized Wavelet
    nl=2*length(h);
    [uu,nuu] = flippeur(h,nu) ; [h,nu] = convoluteur(h,nu,uu,nuu);
    h = h/sqrt(2); [v,nv] = tildeurflippeur(-h,nu);
    x0=2*Nwt; % parameters for the centering of the wavelet
    x0Appro=2*Nwt;
end
[hh1,nh1] = flippeur(h,nu) ;
[gg1,ng1] = flippeur(v,nv) ;

%--- Predict the max # of octaves available given Nwt, and take the min with
nbvoies= fix( log2(length(data)) );
nbvoies = min( fix(log2(n/(nl+3)))  , nbvoies); %   safer, casadestime having problems
nbvoies = min(nbvoies, J2);


% INITIALIZE
[n1,n2] = size(data);
ALH =zeros(n1,n2);
AHL =zeros(n1,n2);
AHH =zeros(n1,n2);
LL =zeros(n1,n2);
HL =zeros(n1,n2);
HH =zeros(n1,n2);
LH =zeros(n1,n2);
tmpcoef=zeros(n1,n2,4);

%--- Compute the WT, calculate statistics
LL=data;
sidata=size(data);
for j=1:nbvoies         % Loop Scales
    
    njtemp = size(LL) ;
    %-- border effect
    fp=length(hh1); % index of first good value
    lp=njtemp; % index of last good value
    
    
    %% WAVELET COEFFICIENTS
    
    %-- OH convolution and subsampling
    OH=conv2(LL,gg1); OH(isnan(OH))=Inf;
    OH(:,1:fp-1)=Inf;
    OH(:,lp(2)+1:end)=Inf;
    OH=OH(:,(1:1:njtemp(2))+x0-1);
    %-- HH convolution and subsampling
    HH=conv2(OH,gg1');HH(isnan(HH))=Inf;
    HH(1:fp-1,:)=Inf;
    HH(lp(1)+1:end,:)=Inf;
    HH=HH((1:1:njtemp(1))+x0-1,:);
    %-- LH convolution and subsampling
    LH=conv2(OH,hh1');LH(isnan(LH))=Inf;
    LH(1:fp-1,:)=Inf;
    LH(lp(1)+1:end,:)=Inf;
    LH=LH((1:1:njtemp(1))+x0Appro-1,:);
    clear OH
    %-- OL convolution and subsampling
    OL=conv2(LL,hh1);OL(isnan(OL))=Inf;
    OL(:,1:fp-1)=Inf;
    OL(:,lp(2)+1:end)=Inf;
    OL=OL(:,(1:1:njtemp(2))+x0Appro-1);
    %-- HL convolution and subsampling
    HL=conv2(OL,gg1');HL(isnan(HL))=Inf;
    HL(1:fp-1,:)=Inf;
    HL(lp(1)+1:end,:)=Inf;
    HL=HL((1:1:njtemp(1))+x0-1,:);
    %-- LL convolution and subsampling
    LL=conv2(OL,hh1');LL(isnan(LL))=Inf;
    LL(1:fp-1,:)=Inf;
    LL(lp(1)+1:end,:)=Inf;
    LL=LL((1:1:njtemp(1))+x0Appro-1,:);
    clear OL
    
    % update wavelet filters for next scale
    [hh1,nh1] = upsample1(hh1,nh1); [gg1,ng1] = upsample1(gg1,ng1);
    
%     %-- passage Norme L1
%     ALH=abs(LH)/2^(j/Norm); AHL=abs(HL)/2^(j/Norm); AHH=abs(HH)/2^(j/Norm);
% %     leaders(j).supcoefnoint=max([max(abs(reshape(ALH(isfinite(ALH)),1,[]))) max(abs(reshape(AHL(isfinite(AHL)),1,[]))) max(abs(reshape(AHH(isfinite(AHH)),1,[])))]);
%     %-- fractional integration by gamma
%     ALH=ALH*2^(gamint*j); AHL=AHL*2^(gamint*j); AHH=AHH*2^(gamint*j);
% %     leaders(j).supcoef=max([max(abs(reshape(ALH(isfinite(ALH)),1,[]))) max(abs(reshape(AHL(isfinite(AHL)),1,[]))) max(abs(reshape(AHH(isfinite(AHH)),1,[])))]);
    
    %-- passage Norme L1 and  fractional integration by gamma
    ALH=abs(LH)/2^(j/Norm)*2^(gamint*j); AHL=abs(HL)/2^(j/Norm)*2^(gamint*j); AHH=abs(HH)/2^(j/Norm)*2^(gamint*j);
    
    %% LEADERS
    
    if j==1
        %-- compute and store leaders sans voisin
        tmpcoef(:,:,1)=ALH;tmpcoef(:,:,2)=AHL;tmpcoef(:,:,3)=AHH;
        leaders(j).sans_voisin.valueall = max(tmpcoef,[],3);
    else
        %-- compute and store leaders sans voisin
        tmpcoef(:,:,1)=ALH;tmpcoef(:,:,2)=AHL;tmpcoef(:,:,3)=AHH; 
        tmpcoef(:,:,4)=leaders(j-1).sans_voisin.valueall;
        leaders(j).sans_voisin.valueall = max(tmpcoef,[],3);
        leaders(j-1).sans_voisin.valueall=[];
    end
    
    leaders(j).valueall = minmaxfilt(leaders(j).sans_voisin.valueall,[1 1]*NHlam*2^(j-1),'max','same'); % 3 lambda
    
    % positions of leaders
    lesx=1:2^0:sidata(2); lesy=1:2^0:sidata(1);
    [i1,j1]=find(isfinite(leaders(j).valueall));
    leaders(j).ypos=lesx(min(j1):max(j1));
    leaders(j).xpos=lesy(min(i1):max(i1));

    leaders(j).value=leaders(j).valueall(min(i1):max(i1),min(j1):max(j1));

    leaders(j).gamma=gamint;
    nj.L(j) = numel(leaders(j).value);
    
end
leaders(j).sans_voisin.valueall=[];

%in case of last scale empty
if isempty(leaders(end).value)
    leaders=leaders(1:end-1);
    nj.L=nj.L(1:end-1);
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INLINE FUNCTIONS
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [a,b]=convoluteur(c,d,e,f);
        a=conv(c,e);
        b=d+f;
    end

    function [a,b]=flippeur(c,d);
        ls=max(size(c));
        b=-ls+1-d;
        a=fliplr(c);
    end

    function [a,b]=upsample1(c,d);
        ls=max(size(c));
        a=zeros(1,2*ls-1);
        a(1:2:2*ls-1)=c;
        b=2*d;
    end

    function [a,b]=tildeurflippeur(c,d);
        ls=max(size(c));
        k=d:1:ls+d-1;
        b=-ls+1-d;
        a=(-1).^(k).*c;
        a=fliplr(a);
    end

    

end % function end
