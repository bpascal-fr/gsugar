
function [data] = FIN2Dpatch(hh,N)


%% function [data] = FIN2Dpatch(hh,N)
%
% 2D synthesis of patches of fractionally integrated Gaussian noise with constant H
%
%% -- input:
%   hh          :   N x N matrix with h values
%   N           :   image size
%% -- output:
%   data        :   N x N image of patches of fractionally integrated Gaussian noise

NwtFI=8;

Hvec=unique(hh);
nbH=length(Hvec);

% Hmed=max(Hvec);
Hmed=mean(Hvec);
aa=Hvec-Hmed; % orders of fractional integration for different patches
xx=randn(N); % random seed
data=zeros(N);
for hid=1:nbH;
    hm=(find(hh==Hvec(hid)));
    ddB=xx;
    ddB = ddB-mean(ddB(hm)); ddB = ddB/std(ddB(hm));
    ddB= fi_2d(ddB,aa(hid),NwtFI);
    data(hm)=ddB(hm);
end

% global integration to generate motion
data = fi_2d(data,Hmed+1,NwtFI);


    function data_out = fi_2d(data_in,alpha,Nwt)
        L = size(data_in);
        L = L(1);
        J = log2(L);
        if Nwt < 100
            h = daubcqf(2*Nwt,'min');
        else
            h = MakeONFilter('Symmlet',Nwt-100);
        end
        wt = mdwt(data_in,h,J);
        deriv = ones(L,L);
        for kj = 1 : J
            deriv(1:2^(J-kj),1:2^(J-kj)) =deriv(1:2^(J-kj),1:2^(J-kj)) * 2^(alpha);
        end
        deriv_wt = deriv .* wt;
        data_out = midwt(deriv_wt,h,J);
    end
end