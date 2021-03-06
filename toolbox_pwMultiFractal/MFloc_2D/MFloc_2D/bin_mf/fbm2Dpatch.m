
function [Xfbm,Xfbm0] = fbm2Dpatch(hh,N,WIN,STDN)


%% function [Xfbm] = fbm2Dpatch(HH,N)
%
% 2D synthesis of patches of fBm with constant H
% After hermine Bierme PhD Thesis, July 2005
% After hermine Bierme routine, July 2005
% Modif : PA, Caracas, July2005
% patches modif: HW, TLS, 092012
%
%% -- input:
%   hh          :   N x N matrix with h values
%   N           :   image size
%% -- output:
%   Xfbm        :   N x N image of patches of fBm


if nargin<3; WIN=1; STDN=0; end
if nargin<4; STDN=0; end
WIN0=WIN; WIN=2*ceil(WIN/2);



Hvec=unique(hh);
nbH=length(Hvec);
Hmask=zeros(nbH,N,N);
for ih=1:nbH
    tmpH=zeros(N); idH=find(hh==Hvec(ih)); tmpH(idH)=1;
    Hmask(ih,:,:)=tmpH;
    if find(idH==1); hBid=ih; end
end
    
    Hmask0=Hmask; Hmask=zeros(nbH,N+WIN,N+WIN);
    Hmask(:,WIN/2+1:end-WIN/2,WIN/2+1:end-WIN/2)=Hmask0;
    Hmask(hBid,1:WIN/2,:)=1;
    Hmask(hBid,:,1:WIN/2)=1;
    Hmask(hBid,end-WIN/2+1:end,:)=1;
    Hmask(hBid,:,end-WIN/2+1:end)=1;
    N0=N+WIN;
    N=2*N0;

    
if WIN0>1
    Wconv=ones(WIN)/WIN^2;
    x=1:N0; [Xip,Yip]=meshgrid(x); 
    IDds=round(WIN/2):WIN:N0;
    X0=Xip(IDds,IDds);
    Y0=Yip(IDds,IDds);
end

Xfbm=zeros(N0);Xfbm0=Xfbm;

if (~isempty(find(Hvec<0))|~isempty(find(Hvec>0.75)))
    error('select 0<H<0.75')
end

%% vecteur normal %%
randn('state',sum(100*clock));
x1=randn(2*N,2*N);
x2=randn(2*N,2*N);

for hid=1:nbH;
    H=Hvec(hid);
    
    a=2*H;
    
    % Calcul de la fonction de covariance %
    Kkl=zeros(N+1,N+1);
    K = [1:1:N+1] ; K = repmat(K,N+1,1) ; K = (K-1)/N ; L = K' ;
    r = K.^2 + L.^2 ;
    index = find(r<1) ;
    Kkl(index) = 1-1/2*a-r(index).^H+1/2*a*r(index) ;
    
    %% P???riodisation de K %%
    P=zeros(2*N,2*N);
    P(1:N+1,1:N+1)=Kkl(1:N+1,1:N+1);
    P(N+2:2*N,1:N+1)=Kkl(N:-1:2,1:N+1);
    P(1:N+1,N+2:2*N)=Kkl(1:N+1,N:-1:2);
    P(N+2:2*N,N+2:2*N)=Kkl(N:-1:2,N:-1:2);
    
    % Valeurs propres de P %
    V=fft2(P);
    % On s'assure que les vap sont positives: vmin doit etre positif %
    vmin=min(min(real(V))');
    
% x1=randn(2*N,2*N);
% x2=randn(2*N,2*N);    
    % Vecteur gaussien admettant P comme matrice de covariance %
    x=x1+i*x2;
    %% vecteur gaussien obtenu %%
    D=x.*sqrt(V);
    y=(2*N)*ifft2(D);
    z1=real(y);
    
    % On obtient un vecteur gaussien de matrice de covariance K %
    %% Il faut rester dans le disque unit??? %%
    M=floor(N/sqrt(2));
    %% On restreint z au carr??? contenu dans le disque unit??? %%
    g=z1(1:M+1,1:M+1);
    
    %% Il faut ajuster pour obtenir une surface Brownienne fractionnaire
    %% On g???n???re 2 va ind???pentes gaussienes centr???es de variance a %%
    X=sqrt(a)*randn(2,1);
    
    %% surface BF %%
    k = [1:1:M+1] ; Z1 = (k-1)*X(1,1)/N ; Z1 = Z1' ; Z1 = repmat(Z1,1,M+1) ;
    j = [1:1:M+1] ; Z2 = (j-1)*X(2,1)/N ; Z2 = repmat(Z2,M+1,1) ;
    Z = g + Z1 + Z2 ;
    
    %% On obtient le MBF en lui imposant d'etre nul en 0 %%
    B=Z-Z(1,1);
%     X=[1:M+1]/(M+1);
    B=B(1:N0,1:N0); 
    B0=B;
    % normalize for visually less striking transition between patches
    if WIN0>1
        % compute local mean in WIN x WIN sliding windows
        tmpc =conv2(B0,Wconv);   
        tmpc = tmpc(WIN:end-(WIN-1),WIN:end-(WIN-1));
        % downsample: only one value per window
        tmpc=tmpc(1:WIN:end,1:WIN:end);
        % interpolate to original size
        i1=1:length(tmpc);
        dd=interp2(X0(i1,i1),Y0(i1,i1),tmpc,Xip,Yip,'*cubic');
        % remove smooth trend
        B=B-dd; 
        dB1 = diff(B) ; 
        dB2 = diff(B') ; 
        dB = sqrt((nanvar(dB1(:))+nanvar(dB2(:)))/2) ; 
        B = B/dB ;

        
        
        % normalize local variance
        if STDN
%             dd1=conv2(B0,Wconv,'same');
%             dd2 =conv2(B0.^2,Wconv,'same')-dd1.^2;

            tmpv =conv2(B0.^2,Wconv,'valid');%tmpv = tmpv(WIN:end-(WIN-1),WIN:end-(WIN-1));
            tmpvds=tmpv(1:WIN:end,1:WIN:end)-tmpc.^2;
            dd2=interp2(X0(i1,i1),Y0(i1,i1),tmpvds,Xip,Yip,'linear');

            if sum(dd2(:)<=0)
                disp('NEGATIVE');
            end
            
            dd2(dd2(:)<=0)=mean(dd2(dd2(:)>0));
            B=B./sqrt(dd2);
        end
    end
    idnn=find(~isnan(B));
    B = B-mean(B(idnn)); % B = B/std(B(idnn));
    hm=find(squeeze(Hmask(hid,:,:)));
    Xfbm(hm)=B(hm);
    
    if WIN0>1
        B=B0;
        idnn=find(~isnan(B));
        B = B-mean(B(idnn)); B = B/std(B(idnn));
        hm=find(squeeze(Hmask(hid,:,:)));
        Xfbm0(hm)=B(hm);
    else
        Xfbm0=Xfbm;
    end
    
end
Xfbm0=Xfbm0(WIN/2+1:end-WIN/2,WIN/2+1:end-WIN/2);
Xfbm=Xfbm(WIN/2+1:end-WIN/2,WIN/2+1:end-WIN/2);

