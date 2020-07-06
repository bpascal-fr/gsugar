% bf_2D
% 2D synthesis of fBm
% After hermine Bierme PhD Thesis, July 2005
% After hermine Bierme routine, July 2005
% Modif : PA, Caracas, July2005

function [B,X,Kkl,r] = bf_2D_patch(H,N,fignumber) ; 
% H = 0.7 ; N = 16 ; fignumber = 10 ; 

if ((H<0)|(H>0.75))
    H=input('prendre 0<H<0.75 H=')
end

a=2*H;
% N=input('pas de discr�tisation N=')

% Calcul de la fonction de covariance %
Kkl=zeros(N+1,N+1);
K = [1:1:N+1] ; K = repmat(K,N+1,1) ; K = (K-1)/N ; L = K' ;  
r = K.^2 + L.^2 ; 
index = find(r<1) ; 
Kkl(index) = 1-1/2*a-r(index).^H+1/2*a*r(index) ;

%% P�riodisation de K %%
P=zeros(2*N,2*N);
P(1:N+1,1:N+1)=Kkl(1:N+1,1:N+1);
P(N+2:2*N,1:N+1)=Kkl(N:-1:2,1:N+1);
P(1:N+1,N+2:2*N)=Kkl(1:N+1,N:-1:2);
P(N+2:2*N,N+2:2*N)=Kkl(N:-1:2,N:-1:2);

% Valeurs propres de P %
V=fft2(P);
% On s'assure que les vap sont positives: vmin doit etre positif %
vmin=min(min(real(V))');


% Vecteur gaussien admettant P comme matrice de covariance %
%% vecteur normal %%
 randn('state',sum(100*clock));
 x1=randn(2*N,2*N);
 x2=randn(2*N,2*N);
 x=x1+i*x2;
 %% vecteur gaussien obtenu %%
 D=x.*sqrt(V);
y=(2*N)*ifft2(D);
 z1=real(y);

 % On obtient un vecteur gaussien de matrice de covariance K %
 %% Il faut rester dans le disque unit� %%
 M=floor(N/sqrt(2));
 %% On restreint z au carr� contenu dans le disque unit� %%
 g=z1(1:M+1,1:M+1);

 %% Il faut ajuster pour obtenir une surface Brownienne fractionnaire
 %% On g�n�re 2 va ind�pentes gaussienes centr�es de variance a %%
 X=sqrt(a)*randn(2,1);

 %% surface BF %%
 k = [1:1:M+1] ; Z1 = (k-1)*X(1,1)/N ; Z1 = Z1' ; Z1 = repmat(Z1,1,M+1) ; 
 j = [1:1:M+1] ; Z2 = (j-1)*X(2,1)/N ; Z2 = repmat(Z2,M+1,1) ;
 Z = g + Z1 + Z2 ; 
 
%  %% surface BF %%
%    for k=1:M+1
%        for j=1:M+1
%            Z(k,j)=g(k,j)+(k-1)*X(1,1)/N+(j-1)*X(2,1)/N;
%        end;
%    end;
%  
 %% On obtient le MBF en lui imposant d'etre nul en 0 %%
 B=Z-Z(1,1);
X=[1:M+1]/(M+1);

if fignumber ~= 0 
    figure(fignumber) ; clf ; 
    mesh(X,X,B)
    title(['FBM H= ',num2str(H)]);
end

%figure(2)
%colormap('bone')
%pcolor(X,X,B)
%title(['FBM H= ',num2str(H)]);
%shading interp
%shading flat
%colorbar('horiz')

%set(gcf,'PaperpositionMode','auto')
%set(gcf,'Renderer','Zbuffer')
% print -depsc 'MBFLabo.eps'