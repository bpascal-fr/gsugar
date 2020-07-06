function [coef, leaders, nj] = DLPx1dloc(data, Nwt, gamint, p, csym, Jmax)
%% function [coef, leaders, nj] = DLPx1dloc(data, Nwt, gamint, p, csym, Jmax)
%
%% INPUT
%   data    - the data :-)
%   Nwt     - number of zero moments of wavelet (Daubechies')
%   gamint  - order of fractional integration
%             [ default: 0 ]
%   p       - p  of wavelet p-leaders
%             if p=Inf, standard wavelet leaders are selected
%             [ default: standard wavelet leaders selected ]
%   csym    - 0:            continuous time dyadic scale wavelet transform (DCWT)
%             1:            continuous time dyadic scale wavelet transform (DCWT) with symmetrized wavelet / scaling function
%             [ default: 1 (DWT) selected ]
%   Jmax    - coarsest scale to be computed
%             [ default: largest available scale ]
%
%% OUTPUT
%   coef(j) - structure containing computed wavelet coefficients dx (L1 norm)
%       .value          absolute values of valid dx
%       .value_noabs    values of valid dx
%       .value_sbord    absolute values of all dx (border effects removed)
%       .nbord          first and last valid coefficient
%       .supcoefnoint   sup(|dx(j,.)|) before fractional integration: h_min estimation
%       .supcoefid      sup(|dx(j,.)|) after fractional integration
%       .supcoef        index of sup(|dx(j,.)|)
%       .gamma          order of fractional integration
%   leaders(j) - structure containing computed wavelet (p-)leaders lx
%       .value          value of valid leaders
%       .value_sbord    value of all leaders (border effects removed)
%       .nbord          first and last valid coefficient
%       .mincoef        inf(|lx(j,.)|) after fractional integration
%       .mincoefid      index of inf(|lx(j,.)|)
%       .supcoefL       sup(|lx(j,.)|)
%       .supcoefidL     index of sup(|lx(j,.)|)
%       .gamma          order of fractional integration
%       .p              p of p-leaders (Inf if leaders)
%       .sans_voisin    structure with same fields .value, .value_sbord and
%                       .nbord described above for leaders with
%                       single \lambda instead of 3\lambda neighborhood
%
% PA/HW, Lyon/Toulouse, August 2012

Norm=1;

if nargout==1; doLeaders=0; else; doLeaders=1; end

if nargin<3; gamint=0; p=Inf; csym=1; Jmax=Inf; end;
if nargin<4; p=Inf; csym=1; Jmax=Inf; end;
if nargin<5; csym=1; Jmax=Inf; end;
if nargin<6; Jmax=Inf; end;

if p==Inf; doL=1; else; doL=0; end;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% continuous time dyadic scale
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%s
u = rlistcoefdaub(Nwt) ; nu = 0 ;
if csym == 0
    % Daubechies Wavelet
    [v,nv] = tildeurflippeur(u,nu);  nv = nv + 1;
elseif csym ==1
    % Daubechies Symmetrized Wavelet
    [uu,nuu] = flippeur(u,nu) ; [u,nu] = convoluteur(u,nu,uu,nuu);
    u = u/sqrt(2); [v,nv] = tildeurflippeur(-u,nu);
end
[h1,nh1] = flippeur(u,nu) ;
[g1,ng1] = flippeur(v,nv) ;
clear nborder

ndata = length(data) ;

J= fix( log2(ndata) );
J= min( fix(log2(ndata/(2*Nwt+1)))  , J);  %   safer, casadestime having problems
J=min(J,Jmax);

nd = 0 ;
wt = zeros(J,ndata) ;  nwt = zeros(1,J+1) ;
appro = zeros(J+1,ndata) ; nappro = zeros(1,J+1) ;
appro(1,:) = data ; nappro(1) = nd ;
hh = h1; nhh = nh1 ;
gg = -g1 ; ngg = ng1 ;
appr = data ; nappr = nd ;

for j=1:1:J
    NU(j)=length(gg);
    [tampon,ntampon] = convoluteur(appr,nappr,gg,ngg) ;
    wt(j,:) = tampon(-ntampon+1:-ntampon+ndata) ; nwt(j) = ntampon ;
    [appr,nappr] = convoluteur(appr,nappr,hh,nhh) ;
    appro(j+1,:) = appr(-nappr+1:-nappr+ndata) ; nappro(j+1) = nappr ;
    [hh,nhh] = upsample1(hh,nhh) ;
    [gg,ngg] = upsample1(gg,ngg) ;
end

jj = [1:1:J] ;
if csym == 0
    gauchetheo = 2.^jj*Nwt-2.^jj+1 ;
    droittheo(1)  =1 ;
    for j=2:1:J
        droittheo(j) = 2^(j-1)*Nwt ;
    end
    droittheo = cumsum(droittheo) ;
    nborder(:,1) = gauchetheo' ;
    nborder(:,2) = droittheo' ;
else
    gauchetheo = cumsum((NU-1)/2)+1;
    droittheo = gauchetheo-1 ;
    nborder(:,1) = gauchetheo' ;
    nborder(:,2) = droittheo' ;
end


for j=1:J
    %% output wavelet coefficients
    decime=wt(j,nborder(j,1):end-nborder(j,2));
    nj.W(j) = length(decime);            % number of coefficients
    AbsdqkW = abs(decime)*2^(j/2)/2^(j/Norm);   %%%%%%% passage Norme L1
    % hmin before integration
    coef(j).supcoefnoint=max(AbsdqkW);
    % fractional integration
    AbsdqkW = AbsdqkW*2^(gamint*j);
    [coef(j).supcoef, coef(j).supcoefid]=max(AbsdqkW);
    coef(j).value=AbsdqkW;
    coef(j).value_noabs=(decime)*2^(j/2)/2^(j/Norm);
    coef(j).value_sbord=wt(j,:)*2^(j/2)/2^(j/Norm)*2^(gamint*j);
    coef(j).nbord=[nborder(j,1), length(wt(j,:))-nborder(j,2)];;
    coef(j).gamma=gamint;
    if doLeaders
        nj.W(j)=length(coef(j).value);
    else
        coef(j).nj=length(coef(j).value);
    end
end

if doLeaders
    if doL;
        %% these are the wavelet leaders mimicking an almost symmetric (3)\lambda neighborhood of the size of that of the dyadic leaders
        WL0=zeros(size(wt));
        for j=1:J
            DEC(j,:)=abs(wt(j,:))*2^(j/2)/2^(j/Norm)*2^(gamint*j);
            if j==1;
                WL0(1,:)=DEC(1,:);
            else
                WL0(j,:)=max([WL0(j-1,:);DEC(j,:)]);
            end
            L_sb(j,:)=minmaxfilt1(WL0(j,:),3*2^(j-1),'max','same');
            L_sb_sv(j,:)=minmaxfilt1(WL0(j,:),2^(j-1),'max','same');
            leaders(j).value_sbord=L_sb(j,:);
            leaders(j).sans_voisin.value_sbord=L_sb_sv(j,:);
            leaders(j).nbord=coef(j).nbord+[+1 -1]*floor(3*2^(j-1)/2) + [-1 0];
            leaders(j).sans_voisin.nbord=coef(j).nbord+[+1 -1]*floor(2^(j-1)/2) + [-1 0];
            leaders(j).value=leaders(j).value_sbord(leaders(j).nbord(1):leaders(j).nbord(2));
            leaders(j).sans_voisin.value=leaders(j).sans_voisin.value_sbord(leaders(j).sans_voisin.nbord(1):leaders(j).sans_voisin.nbord(2));
            nj.L(j)=length(leaders(j).value);
            nj.L_sv(j)=length(leaders(j).sans_voisin.value);
        end
    else
        %% these are the wavelet leaders mimicking an almost symmetric (3)\lambda neighborhood of the size of that of the dyadic leaders
        WL0=zeros(size(wt));
        for j=1:J
            DEC(j,:)= ( abs(wt(j,:))*2^(j/2)/2^(j/Norm)*2^(gamint*j) ).^p;
            if j==1;
                WL0(1,:)=DEC(1,:);
            else
                WL0(j,:)=sum([WL0(j-1,:);DEC(j,:)]);
            end
            FiltreSum=ones(1,3*2^(j-1)); tmpL_sb=conv(WL0(j,:),FiltreSum);
            L_sb(j,:)=tmpL_sb(floor(3*2^(j-1)/2)+1:ndata+floor(3*2^(j-1)/2));
            FiltreSum=ones(1,2^(j-1)); tmpL_sb=conv(WL0(j,:),FiltreSum);
            L_sb_sv(j,:)=tmpL_sb(floor(2^(j-1)/2)+1:ndata+floor(2^(j-1)/2));
            leaders(j).value_sbord=L_sb(j,:);
            leaders(j).sans_voisin.value_sbord=L_sb_sv(j,:);
            leaders(j).nbord=coef(j).nbord+[+1 -1]*floor(3*2^(j-1)/2) + [-1 0];
            leaders(j).sans_voisin.nbord=coef(j).nbord+[+1 -1]*floor(2^(j-1)/2) + [-1 0];
            leaders(j).value=leaders(j).value_sbord(leaders(j).nbord(1):leaders(j).nbord(2));
            leaders(j).sans_voisin.value=leaders(j).sans_voisin.value_sbord(leaders(j).sans_voisin.nbord(1):leaders(j).sans_voisin.nbord(2));
            nj.L(j)=length(leaders(j).value);
            nj.L_sv(j)=length(leaders(j).sans_voisin.value);
        end
        for j=1:J
            leaders(j).sans_voisin.value = (2^(-j).*leaders(j).sans_voisin.value).^(1/p) ;
            leaders(j).value = (2^(-j).*leaders(j).value).^(1/p) ;
            leaders(j).sans_voisin.value_sbord = (2^(-j).*leaders(j).sans_voisin.value_sbord).^(1/p) ;
            leaders(j).value_sbord = (2^(-j).*leaders(j).value_sbord).^(1/p) ;
        end
    end
    %%
    for j=1:J
        leaders(j).gamma=gamint;
        [leaders(j).mincoef, leaders(j).mincoefid]=min(leaders(j).value);
        [leaders(j).supcoefL, leaders(j).supcoefidL]=max(leaders(j).value);
        leaders(j).supcoefnoint=coef(j).supcoefnoint; leaders(j).supcoef=coef(j).supcoef; leaders(j).supcoefid=coef(j).supcoefid;
        leaders(j).p=p;
    end
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

    function h=rlistcoefdaub(regu);
        
        if ~(regu<11 & regu>0),
            mess=[' Ondelette non implantee'];
            disp(mess);
            return;
        else
            if regu==1,
                h=[1/sqrt(2) 1/sqrt(2)];
                %  g=[1/sqrt(2) -1/sqrt(2)];
            end
            if regu==2,
                h(1:2)=[0.482962913145 0.836516303738];
                h(3:4)=[0.224143868042 -0.129409522551];
            end
            if regu==3,
                h(1:2)=[0.332670552950 0.806891509311];
                h(3:4)=[0.459877502118 -0.135011020010];
                h(5:6)=[-0.085441273882 0.035226291882];
            end
            if regu==4,
                h(1:2)=[0.230377813309 0.714846570553];
                h(3:4)=[0.630880767930 -0.027983769417];
                h(5:6)=[-0.187034811719 0.030841381836];
                h(7:8)=[0.032883011667 -0.010597401785];
            end
            if regu==5,
                h(1:2)=[0.160102397974 0.603829269797];
                h(3:4)=[0.724308528438 0.138428145901];
                h(5:6)=[-0.242294887066 -0.032244869585];
                h(7:8)=[0.077571493840 -0.006241490213];
                h(9:10)=[-0.012580751999 0.003335725285];
            end
            if regu==6,
                h(1:2)=[0.111540743350 0.494623890398];
                h(3:4)=[0.751133908021 0.315250351709];
                h(5:6)=[-0.226264693965 -0.129766867567];
                h(7:8)=[0.097501605587 0.027522865530];
                h(9:10)=[-0.031582039318 0.000553842201];
                h(11:12)=[0.004777257511 -0.001077301085];
            end
            if regu==7,
                h(1:2)=[0.077852054085 0.396539319482];
                h(3:4)=[0.729132090846 0.469782287405];
                h(5:6)=[-0.143906003929 -0.224036184994];
                h(7:8)=[0.071309219267 0.080612609151];
                h(9:10)=[-0.038029936935 -0.016574541631];
                h(11:12)=[0.012550998556 0.000429577973];
                h(13:14)=[-0.001801640704 0.000353713800];
            end
            if regu==8,
                h(1:2)=[0.054415842243 0.312871590914];
                h(3:4)=[0.675630736297 0.585354683654];
                h(5:6)=[-0.015829105256 -0.284015542962];
                h(7:8)=[0.000472484574 0.128747426620];
                h(9:10)=[-0.017369301002 -0.044088253931];
                h(11:12)=[0.013981027917 0.008746094047];
                h(13:14)=[-0.004870352993 -0.000391740373];
                h(15:16)=[0.000675449406 -0.000117476784];
            end
            if regu==9,
                h(1:2)=[0.038077947364 0.243834674613];
                h(3:4)=[0.604823123690 0.657288078051];
                h(5:6)=[0.133197385825 -0.293273783279];
                h(7:8)=[-0.096840783223 0.148540749338];
                h(9:10)=[0.030725681479 -0.067632829061];
                h(11:12)=[0.000250947115 0.022361662124];
                h(13:14)=[-0.004723204758 -0.004281503682];
                h(15:16)=[0.001847646883 0.000230385764];
                h(17:18)=[-0.000251963189 0.000039347320];
            end
            if regu==10,
                h(1:2)=[0.026670057901 0.188176800078];
                h(3:4)=[0.52720118932 0.688459039454];
                h(5:6)=[0.281172343661 -0.249846424327];
                h(7:8)=[-0.195946274377 0.127369340336];
                h(9:10)=[0.093057364604 -0.071394147166];
                h(11:12)=[-0.029457536822 0.033212674059];
                h(13:14)=[0.003606553567 -0.010733175483];
                h(15:16)=[0.001395351747 0.001992405295];
                h(17:18)=[-0.000685856695 -0.000116466855];
                h(19:20)=[0.000093588670 -0.000013264203];
            end
        end
    end

end % function end
