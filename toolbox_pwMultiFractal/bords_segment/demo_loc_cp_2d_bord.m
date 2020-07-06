
clear all;

addpath(genpath('/Users/hwendt/Documents/MATLAB/TMP/toolbox_pwMultiFractal/MFloc_2D/MFloc_2D'));
%pth=which('demo_loc_cp_2d.m');id=strfind(pth,'/');pth=pth(1:id(end));addpath(genpath(pth));

N=2^9;

%% MODEL
MODEL=2;        % see SWITCH below
% Parameters for random ellipse
dA0=0.15;       % fraction of total area taken up by object
bord=200;

switch MODEL
    %% FBM - RANDOM ELLIPSE with prescribed area and large border
    case 1
        HH=[0.75, 0.55];
        [hh,bdcrd,dApx,INBORD,usedgeom] = gen_model_hloc(N,HH,dA0,bord);
        C1=HH; C2=zeros(size(HH));
        %% MRW - RANDOM ELLIPSE with prescribed area and large border
    case 2
        %             H=[0.53 0.6];  Lambda=sqrt([0.08 0.01]);
        %             H=[0.6 0.8];  Lambda=sqrt([0.02 0.02]);
        %             H=[0.6 0.8];  Lambda=sqrt([0.005 0.005]);
        

        C1=[0.8 0.7];
        C1=[0.8 0.6];
        C1=[0.8 0.5];
        
%         C2=[-0.005 -0.02];
        C2=[-0.005 -0.05];
%         C2=[-0.005 -0.08];
        
        
        Lambda=sqrt(-C2);
        H=C1+C2;
        
        [MASK] = gen_model_hloc(N,[1 2],dA0,bord);
end


%% GENERATE DATA

RM_WIN=4*16; % window size to remove smooth trend / stick different zones

if MODEL<2
    [data,data0] = fbm2Dpatch(hh,N,RM_WIN);
else
    L=1;[data,data0,C,D] = mrw2D_patch_HW(H,Lambda,N,L,MASK,RM_WIN) ;
end


%% ESTIMATE

Nwt=3;
gamint=1;
symm=1;
J2=3;
SPatch=20;   % Window size for computing Cp

% [leaders, nj] = DCLx2d_lowmem(data, Nwt, gamint,symm, J2);
% [est,valpos] = Sliding_CP_est_2D(leaders,SPatch,1,J2);

% data0=data; data=zeros(size(data)); data(end/2,end/2)=1;
[leaders, nj] = DCLx2d_lowmem_bord(data, Nwt, gamint,symm, J2);
[est,valpos] = Sliding_CP_est_2D_bord(leaders,SPatch,1,J2);

%% PLOTS
figure(100); clf;  imagesc(data); grid on; colormap(bone)
figure(1);clf;     imagesc(est.c1-gamint); grid on; colorbar; title(['c1 = ',num2str(C1)]);
figure(2);clf;     imagesc(est.c2); grid on; colorbar; title(['c2 = ',num2str(C2)]);
caxis([-0.3 0.1]);
figure(3);clf;     imagesc(est.c3); grid on; colorbar; title(['c3 = ',num2str(zeros(size(C1)))]);
figure(4);clf;
for j=1:J2;
    subplot(ceil(J2/2),2,j);
    imagesc(squeeze(est.C1(:,:,j))-gamint*j); grid on; title(['C1(j=',num2str(j),')']);
end
figure(5);clf;
for j=1:J2;
    subplot(ceil(J2/2),2,j);
    imagesc(squeeze(est.C2(:,:,j))); grid on; title(['C2(j=',num2str(j),')']);
end

