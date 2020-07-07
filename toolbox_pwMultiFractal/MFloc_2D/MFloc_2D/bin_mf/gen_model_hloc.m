function [hh,bdcrd,dApx,INBORD,usedgeom] = gen_model_hloc(N,HH,dA0,bordpx,RECT,usrgeom)

%% function [hh,bdcrd,dApx,INBORD,usedgeom] = gen_model_hloc(N,HH,dA0,bordpx,RECT,usrgeom)
% creates image with one constant value inside an ellipse (or rectangle) and another constant value outside
% - either random position/orientation/size with fixed area (default)
% - or user providedposition/orientation/size with fixed area
%
%% INPUT
% N         - image size N x N                                         [required]
% HH        - const values HH(1) and HH(2) for outer / inner region    [required]
% dA0       - fraction of area of object / inner region                [required]
% bordpx    - # border pixels that can not contain object              [DEFAULT: 40]
% RECT      - rectangle instead of ellipse if RECT==1                  [DEFAULT: Ellipse]
%[usrgeom   - if not provided: random position/orientation/size with fixed area
%           - else: [a, b/a, theta, xc, yc] 
%
% NOTE: coordinates are normalized to [ -1 , 1 ]
%
%% OUTPUT
% hh        - image
% bdcrd     - coordinates of object border: x=bdcrd{1}, y=bdcrd{2}
% dApx      - pixel fraction belonging to inner region with value HH(2)
% INBORD    - inner region within prescribed borders (always 1 if random geometry, i.e., no usrgeom provided)
% usedgeom  - parameters of inner region: [a, b/a, theta, xc, yc] 
%               note: (xc,yc) are the center points BEFORE rotation


% check input arguments and set default values
if nargin<3; error('input N, [h1,h2], area');
elseif nargin<4; DO_RND=1;RECT=0;bordpx=40;
elseif nargin<5; DO_RND=1;RECT=0;
elseif nargin<6; DO_RND=1;
else DO_RND=0; a=usrgeom(1);abf=usrgeom(2);b=a*abf;theta=usrgeom(3);xc=usrgeom(4);yc=usrgeom(5); % geometry / position provided
end

% constants
A0=4; A0px=N^2; % total area
bord=bordpx/N;
NevalE=200; % you can tweak the number of evaluation points for the ellipse here

if DO_RND % random position/orientation/size with fixed area
    INBORD=0;
    % constrain random geometry and position
    if RECT
        amax=2-2*bord; bmin=A0*dA0/amax; % rectangle
    else
        amax=sqrt(2)*(2-2*bord); bmin=A0*dA0/amax/pi*4; % ellipse
    end
    dxyc=bmin/2+bord;
    while ~INBORD
        % random draw geometry
        a=(amax-bmin)*rand(1)+bmin;
        theta=pi*rand(1);
        xc=(randn(1)-0.5)*(2-2*dxyc);yc=(randn(1)-0.5)*(2-2*dxyc);
        % make right area
        if RECT;b=A0*dA0/a;else;b=A0*dA0/a/pi*4;end
        % border
        if RECT
            xx=[-1 1 1 -1 -1]*a/2+xc; yy=[-1 -1 1 1 -1]*b/2+yc; % rectangle
        else
            ttev=linspace(0,pi,NevalE); xx=a/2*cos(ttev); yy=(b/2*sqrt(1-(2*xx/a).^2)); xx=[xx,fliplr(xx)]+xc; yy=[yy,-fliplr(yy)]+yc;
        end
        % check if within image: new random geometry if too large
        xxr=xx*cos(theta)+yy*sin(theta); yyr=-xx*sin(theta)+yy*cos(theta);
        mx=min(xxr);Mx=max(xxr);my=min(yyr);My=max(yyr);
        INBORD=mx-bord>-1&Mx+bord<1&my-bord>-1&My+bord<1;
    end
else % user-provided position/orientation/size constrained to fixed area
    % make right area
    if RECT;Atmp=a*b;else;Atmp=pi/4*a*b;end
    a=a*sqrt(A0*dA0/Atmp); b=b*sqrt(A0*dA0/Atmp); 
    % border
    if RECT
        xx=[-1 1 1 -1 -1]*a/2+xc; yy=[-1 -1 1 1 -1]*b/2+yc; % rectangle
    else
        ttev=linspace(0,pi,NevalE); xx=a/2*cos(ttev); yy=(b/2*sqrt(1-(2*xx/a).^2)); xx=[xx,fliplr(xx)]+xc; yy=[yy,-fliplr(yy)]+yc;
    end
    % check if within borders
    xxr=xx*cos(theta)+yy*sin(theta); yyr=-xx*sin(theta)+yy*cos(theta);
    mx=min(xxr);Mx=max(xxr);my=min(yyr);My=max(yyr);
    INBORD=mx-bord>-1&Mx+bord<1&my-bord>-1&My+bord<1;
end
bdcrd{1}=xxr;bdcrd{2}=yyr;usedgeom=[a,b/a,theta,xc,yc]; % for output only

% create image
x=linspace(-1,1,N); [X1,X2]=meshgrid(x);
X1r=X1*cos(theta)-X2*sin(theta);X2r=X1*sin(theta)+X2*cos(theta); % rotate
hh=ones(N)*HH(1);% outer region
if RECT % interior region
    hh(X2r>yc-b/2&X2r<yc+b/2&X1r>xc-a/2&X1r<xc+a/2)=HH(2);dApx=sum(hh(:)==HH(2))/A0px; % rectangle
else
    hh( ((X1r-xc)/a*2).^2+((X2r-yc)/b*2).^2 <1)=HH(2);dApx=sum(hh(:)==HH(2))/A0px; % ellipse
end

if nargout==0
    figure(1); clf;
    imagesc([-1 1],[-1 1],hh); grid on; hold on;
    plot(bdcrd{1},bdcrd{2},'m'); hold off
end

