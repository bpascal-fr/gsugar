
function U=fgn1D_piecewise(N,H,Variance)


% Stripped-down OSSRGF code by SR based on M. Clausel and B. Vedel, 2008
%
%This program simulates a 1D FGN satisfying the scaling property
% X(x)= a^H X(x)
%
% of size 2^N
%
%Authors : M. Clausel and B. Vedel, 2008
%clausel@univ-paris12.fr
%beatrice.vedel@ens-lyon.fr

%% 2011 January

%stephane.roux@ens-lyon.fr

%% Restriction to standard FBM only (HW, LYS, 06/2018)

% M. Clausel: Laboratoire d'Analyse Mathematique et Appliquee
% UMR 8050 du CNRS
% Universite Paris Est,
% 61 Avenue du General de Gaulle
% 94100 Creteil Cedex, France

%B. Vedel: Laboratoire de Physique
%CNRS UMR 5672
%ENS de Lyon
%46, allee d'Italie
%69007 Lyon

%% Modified by B. Pascal

% July, 18th 2018
l1=1;l2=1;M=N-1;

%COORDINATES
X=(-2*2^M:2:2*2^M)/(2^(M+1));
X(2^M+1)=1/2^M;

%NOUVELLE DENSITE SPECTRALE PSI
tau=abs(X);
Hlist = unique(H);
phi = cell(1,length(Hlist));
for iH =1:length(Hlist)
%     phi{iH}=tau.^(Hlist(iH)+(l1+l2)/2);
      phi{iH}=tau.^(Hlist(iH)+(l1)/2);
end



%CONSTRUCTION OF THE FOURIER TRANSFORM OF THE FIELD (WITHOUT RENORMALIZATION)
%W= Fourier transform of the OSSRGF
%Zr and Zi avoid to include additional symmetries in Fourier
Zr=randn(1,2*2^M+1);
Zi=randn(1,2*2^M+1);
Z = Zr + 1i*Zi;

W = cell(1,length(Hlist));
for iH =1:length(Hlist)
    W{iH}=fftshift(fft2(Z))./phi{iH};
end

%CONSTRUCTION OF THE FIELD (FOURIER INVERSE + RENORMALIZATION)
Zp = cell(1,length(Hlist));
U = zeros(size(H));
for iH =1:length(Hlist)
    T=real(ifft2(ifftshift(W{iH})));
    Zp{iH}=T-T(2^M+1);
    Zp{iH}=Zp{iH}(2:end)-Zp{iH}(1:end-1);
    Zp{iH} = Zp{iH}/std(Zp{iH});
    U(H==Hlist(iH)) =  Zp{iH}(H==Hlist(iH));
end
U = sqrt(Variance).*U;




