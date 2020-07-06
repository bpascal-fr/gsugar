%%%%%%%%
%%
%% function 
%%       
%%       [coefs,jm]=cleandwt2d(data,Nwt,norm,jmax);
%%       
%%
%%  Compute the 2D dyadic wavelet transform with choosen normalisation
%%  Use function mdwt from Rice tollbox
%%
%%%%%%%%
%%
%%
%%  input parameters 
%%
%%      data -- signal to analyse (of any size)
%%      Nwt    -- number of null moments of the Daubechies wavelet
%%      norm   -- normalisation of the wavelet transform.
%%                (equal to 'exp' exponent of the prefactor 
%%                |a|^exp in the definition of WT). Default = -1
%%      jmax   -- number of step; default : the maximum
%%                Default = the maximum
%%
%%  output parameters
%%
%%      coefs -- structure of size the number of scale computed with fields 
%%               'x'  containing coefs from HL filters
%%               'y'  containing coefs from LH filters
%%               'xy'  containing coefs from HH filters
%%               'xssbord' same as 'x' exept we are remove coefficients
%%                         polluted by border effects.
%%               'yssbord' same as 'y' exept we are remove coefficients
%%                         polluted by border effects.
%%               'xyssbord' same as 'xy' exept we are remove coefficients
%%                         polluted by border effects.
%%               'appro'  coefficients from LL filter (for the last scale only)
%%               'approssbord'  same as 'appro' withou polluted coefs.
%%
%%      jm    -- number of scale with non polluted DWT coefs.
%%
%%  USAGE EXAMPLES 
%
%   Nwt=3; norm=-1;
%   load lenna
%   [coefs,jm]=cleandwt2d(i1,Nwt,norm);
%   j=3;
%   figure(1);clf;
%   subplot(131);imagesc(coefs(j).x);colormap('gray');
%   subplot(132);imagesc(coefs(j).y);colormap('gray');
%   subplot(133);imagesc(coefs(j).xy);colormap('gray');
%
%%%%%%%%%
%% S.R., ENS-lyon 11/2007
%


function [coefs,jssbord]=cleandwt2d(data,Nwt,norm,jmax);

if nargin < 3
 norm=-1;
end

J = log2(length(data)) ;
if nargin < 4
  jmax=J;
end

decompnorm=-.5;
expo=norm-decompnorm;

h = daubcqf(2*Nwt,'min');
wt = mdwt(data,h,jmax);

for j=1:jmax

  % we normalize the coefs
  coefs(j).x  = [wt( 1:2^(J-j)             , (2^(J-j) +1):2^(J-j+1))] * 2^(j*2*expo);               
  coefs(j).y  = [wt((2^(J-j) +1):2^(J-j+1) , 1:2^(J-j))] * 2^(j*2*expo);
  coefs(j).xy = [wt((2^(J-j) +1):2^(J-j+1) , (2^(J-j) +1):2^(J-j+1))] * 2^(j*2*expo);

  %on enleve les bords
  nbord=ceil((2^j-1)/2^j*(2*Nwt-1));
  if length(coefs(j).x) > nbord
    jssbord=j-1;
    coefs(j).xssb=coefs(j).x(1:end-nbord,1:end-nbord);
    coefs(j).yssb=coefs(j).y(1:end-nbord,1:end-nbord);
    coefs(j).xyssb=coefs(j).xy(1:end-nbord,1:end-nbord);
  end
end

coefs(j).appro=[wt(1:2^(J-j), 1:2^(J-j))] * 2^(j*2*expo);
coefs(j).approssb=coefs(j).appro(1:end-nbord,1:end-nbord);
