
function [coef, leaders, nj] = DxLx2d(data,Nwt, gamint);
%% function [coef, leaders, nj] = DxLx2d(data,Nwt, gamint);
%%%
%
%         data -- data 2D
%         Nwt  -- wave order
%         norm -- normalisation of the wavelet (a^norm \psi(b-t)/a)
%         jmax -- scale max
%         option = 0 avec bords
%                    sinon sans bords
%
% Herwig Wendt, Stephane G. Roux, Lyon, 2006 - 2008

if nargin<3; gamint=0; end;

norm=-1;
option=1;
jmax=log2(length(data));

% Compute DWT
[coefs,jssbord]=cleandwt2d(data,Nwt,norm,jmax);
%[coefs,jssbord]=cleandwt2d_noabs(data,Nwt,norm,jmax);

J = log2(length(data)) ;

if option == 0
    tagx='x';
    tagy='y';
    tagxy='xy';
else
    tagx='xssb';
    tagy='yssb';
    tagxy='xyssb';
    jmax=jssbord;
end

% computes leaders
for kj = 1:jmax   % boucle sur echelles
    x=getfield(coefs(kj),tagx);
    y=getfield(coefs(kj),tagy);
    xy=getfield(coefs(kj),tagxy);
    
    %% HW 03/09/2008 -->
    x(isnan(x))=Inf;
    y(isnan(y))=Inf;
    xy(isnan(xy))=Inf;
    
%     x(isnan(x))=0;
%     y(isnan(y))=0;
%     xy(isnan(xy))=0;
    %% <--
    
%     keyboard
    
    %% ----> ADDED 16/04/2007
    % hmin
    coef(kj).supcoefnoint=max(abs([reshape(x(find(x<Inf)),1,[]) reshape(y(find(y<Inf)),1,[]) reshape(xy(find(xy<Inf)),1,[])]));
    coef(kj).supcoefnoint_x=max(abs(reshape(x(find(x<Inf)),1,[])));
    coef(kj).supcoefnoint_y=max(abs(reshape(y(find(y<Inf)),1,[])));
    coef(kj).supcoefnoint_xy=max(abs(reshape(xy(find(xy<Inf)),1,[])));
    % fractional integration by gamma
    coef(kj).gamma=gamint;
    x=x*2^(gamint*kj);
    y=y*2^(gamint*kj);
    xy=xy*2^(gamint*kj);
    coef(kj).supcoef=max(abs([reshape(x(find(x<Inf)),1,[]) reshape(y(find(y<Inf)),1,[]) reshape(xy(find(xy<Inf)),1,[])]));
    coef(kj).supcoef_x=max(abs(reshape(x(find(x<Inf)),1,[])));
    coef(kj).supcoef_y=max(abs(reshape(y(find(y<Inf)),1,[])));
    coef(kj).supcoef_xy=max(abs(reshape(xy(find(xy<Inf)),1,[])));
    %% <---- ADDED 16/04/2007
    
    % if nargout>1;
    coef(kj).x=x;
    coef(kj).y=y;
    coef(kj).xy=xy;
    coef(kj).vector.x=reshape(x,1,[]);
    coef(kj).vector.y=reshape(y,1,[]);
    coef(kj).vector.xy=reshape(xy,1,[]);
    coef(kj).vector.all=[coef(kj).vector.x coef(kj).vector.y coef(kj).vector.xy];
    % end
    %nj.W(kj)=length(coef(kj).x);
    nj.W(kj)=length(coef(kj).vector.all);
    Lim(kj)=length(x);  % get size of image
    if kj==1  % les leaders sans voisins sonts les coeff
        leaders(kj).sans_voisin.value.x = abs(x);
        leaders(kj).sans_voisin.value.y = abs(y);
        leaders(kj).sans_voisin.value.xy= abs(xy);
    else      % ont calcule les leaders sans voisins
        DiffL=floor((Lim(kj-1)-2*Lim(kj))/2);

        if 1 % NEW: Replace Loop by matrix operations
            clear temp*
            temp1x=abs(x); [a,b]=size(temp1x);
            temp2x=leaders(kj-1).sans_voisin.value.x(1+DiffL:2:end,1+DiffL:2:end); temp2x=temp2x(1:a, 1:b);
            temp3x=leaders(kj-1).sans_voisin.value.x(2+DiffL:2:end,1+DiffL:2:end); temp3x=temp3x(1:a, 1:b);
            temp4x=leaders(kj-1).sans_voisin.value.x(1+DiffL:2:end,2+DiffL:2:end); temp4x=temp4x(1:a, 1:b);
            temp5x=leaders(kj-1).sans_voisin.value.x(2+DiffL:2:end,2+DiffL:2:end); temp5x=temp5x(1:a, 1:b);
            tempx(1,:,:)=temp1x; tempx(2,:,:)=temp2x; tempx(3,:,:)=temp3x; tempx(4,:,:)=temp4x; tempx(5,:,:)=temp5x;
            leaders(kj).sans_voisin.value.x=squeeze(max(tempx));

            clear temp*
            temp1y=abs(y); [a,b]=size(temp1y);
            temp2y=leaders(kj-1).sans_voisin.value.y(1+DiffL:2:end,1+DiffL:2:end); temp2y=temp2y(1:a, 1:b);
            temp3y=leaders(kj-1).sans_voisin.value.y(2+DiffL:2:end,1+DiffL:2:end); temp3y=temp3y(1:a, 1:b);
            temp4y=leaders(kj-1).sans_voisin.value.y(1+DiffL:2:end,2+DiffL:2:end); temp4y=temp4y(1:a, 1:b);
            temp5y=leaders(kj-1).sans_voisin.value.y(2+DiffL:2:end,2+DiffL:2:end); temp5y=temp5y(1:a, 1:b);
            tempy(1,:,:)=temp1y; tempy(2,:,:)=temp2y; tempy(3,:,:)=temp3y; tempy(4,:,:)=temp4y; tempy(5,:,:)=temp5y;
            leaders(kj).sans_voisin.value.y=squeeze(max(tempy));

            clear temp*
            temp1xy=abs(xy); [a,b]=size(temp1xy);
            temp2xy=leaders(kj-1).sans_voisin.value.xy(1+DiffL:2:end,1+DiffL:2:end); temp2xy=temp2xy(1:a, 1:b);
            temp3xy=leaders(kj-1).sans_voisin.value.xy(2+DiffL:2:end,1+DiffL:2:end); temp3xy=temp3xy(1:a, 1:b);
            temp4xy=leaders(kj-1).sans_voisin.value.xy(1+DiffL:2:end,2+DiffL:2:end); temp4xy=temp4xy(1:a, 1:b);
            temp5xy=leaders(kj-1).sans_voisin.value.xy(2+DiffL:2:end,2+DiffL:2:end); temp5xy=temp5xy(1:a, 1:b);
            tempxy(1,:,:)=temp1xy; tempxy(2,:,:)=temp2xy; tempxy(3,:,:)=temp3xy; tempxy(4,:,:)=temp4xy; tempxy(5,:,:)=temp5xy;
            leaders(kj).sans_voisin.value.xy=squeeze(max(tempxy));

        else
            for kx = 1:length(x)
                for ky = 1:length(y)                                                                % ! length(y) ?
                    % Leader without neighbor scale J:
                    % maximum of 4 neighboring Leaders on scale (J-1) and new coefficient
                    % en x
                    clear temp*
                    temp1=abs(x(kx,ky));
                    % les leaders a l'echelle inferieure
                    %           temp2=leaders(kj-1).sans_voisin.value.x(2*kx-1,2*ky-1);
                    %           temp3=leaders(kj-1).sans_voisin.value.x(2*kx,2*ky-1);
                    %           temp4=leaders(kj-1).sans_voisin.value.x(2*kx-1,2*ky);
                    %           temp5=leaders(kj-1).sans_voisin.value.x(2*kx,2*ky);

                    temp2=leaders(kj-1).sans_voisin.value.x(2*kx-1+DiffL,2*ky-1+DiffL);
                    temp3=leaders(kj-1).sans_voisin.value.x(2*kx+DiffL,2*ky-1+DiffL);
                    temp4=leaders(kj-1).sans_voisin.value.x(2*kx-1+DiffL,2*ky+DiffL);
                    temp5=leaders(kj-1).sans_voisin.value.x(2*kx+DiffL,2*ky+DiffL);
                    leaders(kj).sans_voisin.value.x(kx,ky) = max([temp1 temp2 temp3 temp4 temp5]);

                    % en y
                    clear temp*
                    temp1=abs(y(kx,ky));
                    % les leaders a l'echelle inferieure
                    %           temp2=leaders(kj-1).sans_voisin.value.y(2*kx-1,2*ky-1);
                    %           temp3=leaders(kj-1).sans_voisin.value.y(2*kx,2*ky-1);
                    %           temp4=leaders(kj-1).sans_voisin.value.y(2*kx-1,2*ky);
                    %           temp5=leaders(kj-1).sans_voisin.value.y(2*kx,2*ky);

                    temp2=leaders(kj-1).sans_voisin.value.y(2*kx-1+DiffL,2*ky-1+DiffL);
                    temp3=leaders(kj-1).sans_voisin.value.y(2*kx+DiffL,2*ky-1+DiffL);
                    temp4=leaders(kj-1).sans_voisin.value.y(2*kx-1+DiffL,2*ky+DiffL);
                    temp5=leaders(kj-1).sans_voisin.value.y(2*kx+DiffL,2*ky+DiffL);
                    leaders(kj).sans_voisin.value.y(kx,ky) = max([temp1 temp2 temp3 temp4 temp5]);

                    % en xy
                    clear temp*
                    temp1=abs(xy(kx,ky));
                    % les leaders a l'echelle inferieure
                    %           temp2=leaders(kj-1).sans_voisin.value.xy(2*kx-1,2*ky-1);
                    %           temp3=leaders(kj-1).sans_voisin.value.xy(2*kx,2*ky-1);
                    %           temp4=leaders(kj-1).sans_voisin.value.xy(2*kx-1,2*ky);
                    %           temp5=leaders(kj-1).sans_voisin.value.xy(2*kx,2*ky);

                    temp2=leaders(kj-1).sans_voisin.value.xy(2*kx-1+DiffL,2*ky-1+DiffL);
                    temp3=leaders(kj-1).sans_voisin.value.xy(2*kx+DiffL,2*ky-1+DiffL);
                    temp4=leaders(kj-1).sans_voisin.value.xy(2*kx-1+DiffL,2*ky+DiffL);
                    temp5=leaders(kj-1).sans_voisin.value.xy(2*kx+DiffL,2*ky+DiffL);
                    leaders(kj).sans_voisin.value.xy(kx,ky) = max([temp1 temp2 temp3 temp4 temp5]);
                end
            end

        end % if 0

    end
    KEEPBORD=0;
    leaders(kj).sans_voisin.vector.x=reshape(leaders(kj).sans_voisin.value.x,1,[]);
    leaders(kj).sans_voisin.vector.y=reshape(leaders(kj).sans_voisin.value.y,1,[]);
    leaders(kj).sans_voisin.vector.xy=reshape(leaders(kj).sans_voisin.value.xy,1,[]);
    % on prend le max sur les 8 voisins i.e. 9 coeffs
    %% x
    clear temp*;
    ls.x = zeros(2+length(x),2+length(y));
    ls.x(2:end-1,2:end-1) = leaders(kj).sans_voisin.value.x;
    temp.x(:,:,1) = ls.x(1:end-2,1:end-2);
    temp.x(:,:,2) = ls.x(1:end-2,2:end-1);
    temp.x(:,:,3) = ls.x(1:end-2,3:end);
    temp.x(:,:,4) = ls.x(2:end-1,1:end-2);
    temp.x(:,:,5) = ls.x(2:end-1,2:end-1);
    temp.x(:,:,6) = ls.x(2:end-1,3:end);
    temp.x(:,:,7) = ls.x(3:end,1:end-2);
    temp.x(:,:,8) = ls.x(3:end,2:end-1);
    temp.x(:,:,9) = ls.x(3:end,3:end);
    if KEEPBORD
        leaders(kj).value.x = max(temp.x,[],3);
    else
        temp.max=max(temp.x,[],3);
        leaders(kj).value.x = temp.max(2:end-1, 2:end-1);
    end

    %% y
    clear temp*;
    ls.y = zeros(2+length(x),2+length(y));
    ls.y(2:end-1,2:end-1) = leaders(kj).sans_voisin.value.y;
    temp.y(:,:,1) = ls.y(1:end-2,1:end-2);
    temp.y(:,:,2) = ls.y(1:end-2,2:end-1);
    temp.y(:,:,3) = ls.y(1:end-2,3:end);
    temp.y(:,:,4) = ls.y(2:end-1,1:end-2);
    temp.y(:,:,5) = ls.y(2:end-1,2:end-1);
    temp.y(:,:,6) = ls.y(2:end-1,3:end);
    temp.y(:,:,7) = ls.y(3:end,1:end-2);
    temp.y(:,:,8) = ls.y(3:end,2:end-1);
    temp.y(:,:,9) = ls.y(3:end,3:end);
    if KEEPBORD
        leaders(kj).value.y = max(temp.y,[],3);
    else
        temp.max=max(temp.y,[],3);
        leaders(kj).value.y = temp.max(2:end-1, 2:end-1);
    end

    %% xy
    clear temp*;
    ls.xy = zeros(2+length(x),2+length(y));
    ls.xy(2:end-1,2:end-1) = leaders(kj).sans_voisin.value.xy;
    temp.xy(:,:,1) = ls.xy(1:end-2,1:end-2);
    temp.xy(:,:,2) = ls.xy(1:end-2,2:end-1);
    temp.xy(:,:,3) = ls.xy(1:end-2,3:end);
    temp.xy(:,:,4) = ls.xy(2:end-1,1:end-2);
    temp.xy(:,:,5) = ls.xy(2:end-1,2:end-1);
    temp.xy(:,:,6) = ls.xy(2:end-1,3:end);
    temp.xy(:,:,7) = ls.xy(3:end,1:end-2);
    temp.xy(:,:,8) = ls.xy(3:end,2:end-1);
    temp.xy(:,:,9) = ls.xy(3:end,3:end);
    if KEEPBORD
        leaders(kj).value.xy = max(temp.xy,[],3);
    else
        temp.max=max(temp.xy,[],3);
        leaders(kj).value.xy = temp.max(2:end-1, 2:end-1);
    end

    clear temp*;
    tempx=reshape(leaders(kj).value.x,1,[]);
    tempy=reshape(leaders(kj).value.y,1,[]);
    tempxy=reshape(leaders(kj).value.xy,1,[]);
    leaders(kj).vector.x =tempx;
    leaders(kj).vector.y =tempy;
    leaders(kj).vector.xy=tempxy;
    leaders(kj).vector.all =  [tempx tempy tempxy] ;
    leaders(kj).vector.max = max(max(tempx, tempy), tempxy) ;   % maximum of x,y,xy at each point

    leaders(kj).gamma=gamint;
%     leaders(kj).supcoef=coef(kj).supcoef;
%     leaders(kj).supcoefnoint=coef(kj).supcoefnoint;
%     leaders(kj).mincoef=min(leaders(kj).vector.max);
%     leaders(kj).mincoef_x=min(leaders(kj).vector.x);
%     leaders(kj).mincoef_y=min(leaders(kj).vector.y);
%     leaders(kj).mincoef_xy=min(leaders(kj).vector.xy);
%     leaders(kj).supcoefL=max(leaders(kj).vector.max);
%     leaders(kj).supcoefL_x=max(leaders(kj).vector.x); leaders(kj).supcoefL_y=max(leaders(kj).vector.y); leaders(kj).supcoefL_xy=max(leaders(kj).vector.xy);
%     coef(kj).mincoef=leaders(kj).mincoef; coef(kj).mincoef_x=leaders(kj).mincoef_x;  coef(kj).mincoef_y=leaders(kj).mincoef_y;  coef(kj).mincoef_xy=leaders(kj).mincoef_xy;
%     coef(kj).supcoefL=leaders(kj).supcoefL; coef(kj).supcoefL_x=leaders(kj).supcoefL_x;  coef(kj).supcoefL_y=leaders(kj).supcoefL_y;  coef(kj).supcoefL_xy=leaders(kj).supcoefL_xy;
    %nj.L(kj)=length(leaders(kj).value.x);
    nj.L(kj)=length(leaders(kj).vector.max);
    %nj.L(kj)=length(leaders(kj).vector.all);
    clear temp*

end


%% HW for rockmore: sort out Inf 
% Lyon, 03/09/2008
clear nj; 
for j = 1:jmax
    leaders(j).vector.x=leaders(j).vector.x(find(leaders(j).vector.x<Inf));
    leaders(j).vector.y=leaders(j).vector.y(find(leaders(j).vector.y<Inf));
    leaders(j).vector.xy=leaders(j).vector.xy(find(leaders(j).vector.xy<Inf));
    leaders(j).vector.all=leaders(j).vector.all(find(leaders(j).vector.all<Inf));
    leaders(j).vector.max=leaders(j).vector.max(find(leaders(j).vector.max<Inf));
    if length(leaders(j).vector.max)>0
        nj.L(j)=length(leaders(j).vector.max);
    end
    coef(j).vector.x=coef(j).vector.x(find(coef(j).vector.x<Inf));
    coef(j).vector.y=coef(j).vector.y(find(coef(j).vector.y<Inf));
    coef(j).vector.xy=coef(j).vector.xy(find(coef(j).vector.xy<Inf));
    coef(j).vector.all=coef(j).vector.all(find(coef(j).vector.all<Inf));
    if length(coef(j).vector.all)>0
        nj.W(j)=length(coef(j).vector.all);
    end
    
    %% MINCOEF
    leaders(j).supcoef=coef(j).supcoef;
    leaders(j).supcoefnoint=coef(j).supcoefnoint;
    leaders(j).mincoef=min(leaders(j).vector.max);
    leaders(j).mincoef_x=min(leaders(j).vector.x);
    leaders(j).mincoef_y=min(leaders(j).vector.y);
    leaders(j).mincoef_xy=min(leaders(j).vector.xy);
    leaders(j).supcoefL=max(leaders(j).vector.max);
    leaders(j).supcoefL_x=max(leaders(j).vector.x); leaders(j).supcoefL_y=max(leaders(j).vector.y); leaders(j).supcoefL_xy=max(leaders(j).vector.xy);
    coef(j).mincoef=leaders(j).mincoef; coef(j).mincoef_x=leaders(j).mincoef_x;  coef(j).mincoef_y=leaders(j).mincoef_y;  coef(j).mincoef_xy=leaders(j).mincoef_xy;
    coef(j).supcoefL=leaders(j).supcoefL; coef(j).supcoefL_x=leaders(j).supcoefL_x;  coef(j).supcoefL_y=leaders(j).supcoefL_y;  coef(j).supcoefL_xy=leaders(j).supcoefL_xy;
    
end
    