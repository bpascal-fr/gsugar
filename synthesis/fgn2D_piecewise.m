
function U=fgn2D_piecewise(N,H,Variance)


% Stripped-down OSSRGF code by SR based on M. Clausel and B. Vedel, 2008
%
%This program simulates a 2D FBM satisfying the scaling property
% X(x)= a^H X(x)
%
% of size (2^N x 2^N)
%
%Authors : M. Clausel and B. Vedel, 2008
%clausel@univ-paris12.fr
%beatrice.vedel@ens-lyon.fr
%% 2011 January
%stephane.roux@ens-lyon.fr
%
%% Restriction to standard FBM only (HW, LYS, 06/2018)

%M. Clausel: Laboratoire d'Analyse Math<E9>matique et Appliqu<E9>e
% UMR 8050 du CNRS
% Universit<E9> Paris Est,
% 61 Avenue du G<E9>n<E9>ral de Gaulle
%94100 Cr<E9>teil Cedex, France

%B. Vedel: Laboratoire de Physique
%CNRS UMR 5672
%ENS de Lyon
%46, all<E9>e d'Italie
%69007 Lyon


l1=1;l2=1;M=N-1;
if H >= 1
    Zp=[]; fprintf('H must be (strictrly) between 0 and 1'); return
end


%COORDINATES
X=(-2*2^M:2:2*2^M)/(2^(M+1));
X(2^M+1)=1/2^M;
Y=(-2*2^M:2:2*2^M)/(2^(M+1));
Y(2^M+1)=1/2^M;
XX=X(ones(1,2*2^M+1),:);
YY=Y(ones(1,2*2^M+1),:)';

clear X Y
%rho is the classical pseudonorm associated to the diagonal matrix with eigenvalues  l1 et l2:
%rho(x,y)=(abs(x)^(2/l1) + abs(y)^(2/l2) )^(1/2)
rho=sqrt(abs(XX).^(2/l1)+abs(YY).^(2/l2));
U=rho.^(-l1).*XX;
V=rho.^(-l2).*YY;
Geval=sqrt((abs(U)).^(2/l1)+(abs(V)).^(2/l2));

%NOUVELLE DENSITE SPECTRALE PSI
tau=Geval.*rho;

Hlist = unique(H);
for iH =1:length(Hlist)
    phi{iH}=tau.^(Hlist(iH)+(l1+l2)/2);
end
clear rho tau

%CONSTRUCTION OF THE FOURIER TRANSFORM OF THE FIELD (WITHOUT RENORMALIZATION)
%W= Fourier transform of the OSSRGF
%Zr and Zi avoid to include additional symmetries in Fourier
Zr=randn(2*2^M+1,2*2^M+1);
Zi=randn(2*2^M+1,2*2^M+1);
Z = Zr + 1i*Zi;
for iH =1:length(Hlist)
    W{iH}=fftshift(fft2(Z))./phi{iH};
end
clear Z phi

%CONSTRUCTION OF THE FIELD (FOURIER INVERSE + RENORMALIZATION)
U = zeros(size(H));
for iH =1:length(Hlist)
    T=real(ifft2(ifftshift(W{iH})));
    Zp{iH}=T-T(2^M+1,2^M+1);
    Zh = 1/2*Zp{iH}(:,1:end-1) - 1/2*Zp{iH}(:,2:end);
    Zv = 1/2*Zp{iH}(1:end-1,:) - 1/2*Zp{iH}(2:end,:);
    Zv = 1/2*Zp{iH}(1:end-1,:) - 1/2*Zp{iH}(2:end,:);
    Zp{iH} = 2*Zv(1:2^N,1:2^N);% + Zv(1:2^N,1:2^N);
    Zp{iH} = Zp{iH}/std2(Zp{iH});
    U(H==Hlist(iH)) =  Zp{iH}(H==Hlist(iH));
end


U = sqrt(Variance).*U;





