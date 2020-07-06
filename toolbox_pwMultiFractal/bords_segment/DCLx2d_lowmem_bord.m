function [coefs,leaders,nj] = DCLx2d_lowmem_bord(data, Nwt, gamint,symm, J2, NH_DYAD)

%
%  [coef, leaders, nj] = DCLx2d_lowmem_bord(data, Nwt, gamint,symm, J2)
%
% compute 2D Dyadic Scale Continuous Time wavelet and leader coefficients
% with symmetric border conditions
% outputs as many coefficients as input pixels
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
% HW, IRIT-TLS, 05/2017
% HW, IRIT-TLS, 02/2014
%   following
% SR, ens-Lyon, 11/2013
%%%%%%%%%%

Norm=1;


NHlam=1; % 1 LAMBDA
%NHlam=3; % 3 LAMBDA

%NH_DYAD=1; % use dyadic neighborhood
%NH_DYAD=0; % use constant neighborhood

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

sidata0=size(data);

% for symmetric boundary conditions
PADIT=1;
if symm
    PADVAL=2^(nbvoies-1)*(length(hh1)-1);
else
    PADVAL=2^(nbvoies)*(length(hh1)-1);
end
PADVAL=2*PADVAL;PADVAL=min(PADVAL,min(sidata0));
if PADIT
    data=[fliplr(data(:,1:PADVAL)),data,fliplr(data(:,end-PADVAL+1:end))];
    data=[flipud(data(1:PADVAL,:));data;flipud(data(end-PADVAL+1:end,:))];
else
    PADVAL=0;
end

% for centering
Nhold=nh1; Cumnh=0; Ngold=ng1; Cumng=0;

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

%keyboard

for j=1:nbvoies         % Loop Scales
    SPosh=sum(Cumnh);
    SPosg=sum(Cumng);
    
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
    % size(HH)
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
    Cumnh=[Cumnh, Cumnh(end)+Nhold]; Nhold=nh1;
    Cumng=[Cumng, Cumng(end)+Ngold]; Ngold=ng1;
    
%     %-- passage Norme L1
%     ALH=abs(LH)/2^(j/Norm); AHL=abs(HL)/2^(j/Norm); AHH=abs(HH)/2^(j/Norm);
% %     leaders(j).supcoefnoint=max([max(abs(reshape(ALH(isfinite(ALH)),1,[]))) max(abs(reshape(AHL(isfinite(AHL)),1,[]))) max(abs(reshape(AHH(isfinite(AHH)),1,[])))]);
%     %-- fractional integration by gamma
%     ALH=ALH*2^(gamint*j); AHL=AHL*2^(gamint*j); AHH=AHH*2^(gamint*j);
% %     leaders(j).supcoef=max([max(abs(reshape(ALH(isfinite(ALH)),1,[]))) max(abs(reshape(AHL(isfinite(AHL)),1,[]))) max(abs(reshape(AHH(isfinite(AHH)),1,[])))]);
    
    coefs(j).valueall(:,:,1)=LH/2^(j/Norm)*2^(gamint*j);
    coefs(j).valueall(:,:,2)=HL/2^(j/Norm)*2^(gamint*j);
    coefs(j).valueall(:,:,3)=HH/2^(j/Norm)*2^(gamint*j);

    %-- passage Norme L1 and  fractional integration by gamma
    ALH=abs(LH)/2^(j/Norm)*2^(gamint*j); AHL=abs(HL)/2^(j/Norm)*2^(gamint*j); AHH=abs(HH)/2^(j/Norm)*2^(gamint*j);
    
    %% LEADERS
    
    clear tmpcoef;
    if j==1
        %-- compute and store leaders sans voisin
        tmpcoef(:,:,1)=ALH;tmpcoef(:,:,2)=AHL;tmpcoef(:,:,3)=AHH;
        leaders(j).sans_voisin.valueall = max(tmpcoef,[],3);
    else
        %-- compute and store leaders sans voisin
        ALH=ALH(-SPosh+1:end,-SPosg+1:end); % x:g - y:h
        AHL=AHL(-SPosg+1:end,-SPosh+1:end); % x:h - y:g
        AHH=AHH(-SPosg+1:end,-SPosg+1:end); % x:h - y:h
        nn(1,:)=size(ALH);nn(2,:)=size(AHL);nn(3,:)=size(AHH); nn=min(nn); 
        ALH=ALH(1:nn(1),1:nn(2));AHL=AHL(1:nn(1),1:nn(2));AHH=AHH(1:nn(1),1:nn(2));
        
        tmpcoef(:,:,1)=ALH; % x:g - y:h
        tmpcoef(:,:,2)=AHL; % x:h - y:g
        tmpcoef(:,:,3)=AHH; % x:h - y:h
        tmpcoef(:,:,4)=leaders(j-1).sans_voisin.valueall(1:nn(1),1:nn(2));
        
        leaders(j).sans_voisin.valueall = max(tmpcoef,[],3);
        leaders(j-1).sans_voisin.valueall=[];
    end
    
%    coefs(j).valueall=tmpcoef(:,:,1:3);
    
    if NH_DYAD
        leaders(j).valueall = minmaxfilt(leaders(j).sans_voisin.valueall,[1 1]*NHlam*2^(j-1),'max','same'); % 3 lambda
    else
        leaders(j).valueall = minmaxfilt(leaders(j).sans_voisin.valueall,[1 1]*NHlam,'max','same'); % 3 lambda
    end
    
    % positions of leaders
    lesx=1:2^0:sidata(2); lesy=1:2^0:sidata(1);
    lesx=lesx-PADVAL;lesy=lesy-PADVAL;
    [i1,j1]=find(isfinite(leaders(j).valueall));
    leaders(j).xpos=lesy(min(i1):max(i1));
    leaders(j).ypos=lesx(min(j1):max(j1));
    leaders(j).value=leaders(j).valueall(min(i1):max(i1),min(j1):max(j1));

    leaders(j).gamma=gamint;
    nj.L(j) = numel(leaders(j).value);
    
    lesx=1:2^0:sidata(2); lesy=1:2^0:sidata(1);
    lesx=lesx-PADVAL;lesy=lesy-PADVAL;
    [i1,j1]=find(isfinite(squeeze(sum(coefs(j).valueall,3))));
    coefs(j).xpos=lesy(min(i1):max(i1));
    coefs(j).ypos=lesx(min(j1):max(j1));
    coefs(j).value=coefs(j).valueall(min(i1):max(i1),min(j1):max(j1),:);
    
end
leaders(j).sans_voisin.valueall=[];

%in case of last scale empty
if isempty(leaders(end).value)
    leaders=leaders(1:end-1);
    nj.L=nj.L(1:end-1);
end

% trim
for j=1:length(leaders)
%    [leaders(j).xpos(1), leaders(j).xpos(end); leaders(j).ypos(1), leaders(j).ypos(end)]
    idx=find(leaders(j).xpos>0&leaders(j).xpos<=sidata0(1));
    leaders(j).value=leaders(j).value(idx,:);
    idy=find(leaders(j).ypos>0&leaders(j).ypos<=sidata0(2));
    leaders(j).value=leaders(j).value(:,idy);
    nj.L(j) = numel(leaders(j).value);
    leaders(j).xpos=leaders(j).xpos(idx);
    leaders(j).ypos=leaders(j).ypos(idy);
    
    idx=find(coefs(j).xpos>0&coefs(j).xpos<=sidata0(1));
    coefs(j).value=coefs(j).value(idx,:,:);
    idy=find(coefs(j).ypos>0&coefs(j).ypos<=sidata0(2));
    coefs(j).value=coefs(j).value(:,idy,:);
    nj.W(j) = numel(coefs(j).value);
    coefs(j).xpos=coefs(j).xpos(idx);
    coefs(j).ypos=coefs(j).ypos(idy);
    
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
