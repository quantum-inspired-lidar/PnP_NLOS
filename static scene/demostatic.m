
% Reconstruction procedures for "Plug-and-Play Algorithms for Dynamic Non-line-of-sight Imaging"
% by Juntian Ye, Yu Hong, Xiongfei Su, Xin Yuan, and Feihu Xu.

%% Run this demo script to step through the reconstruction procedures
% Assumes MATLAB R2022b or higher (lower versions may work but have not
% been tested).
addpath('../');
addpath('../fcns/');


%% setting
N = 128;                  % pixel number of x,y
bin = 512;                % pixel number of time or z axis
frame = 1;                % static scene
samp = 16;                % sampling points   (samp x samp)
bin_resolution = 32e-12;            % Time resolution
wall_size = 2;        
mask = definemask(N,samp,frame,bin);   % down sampling operator
width = wall_size/2;                % scan range -width to width (unit:m)
c = 3*10^8;                         % speed of light

% The Light-cone-transform code originally come from: https://www.nature.com/articles/nature25489
[mtx,mtxi] = resamplingOperator(bin);
mtx = gpuArray(single(full(mtx)));
mtxi = gpuArray(single(full(mtxi)));
psf = single(definePsf(N,width,bin,bin_resolution,c));    
fprintf('=================1 setting done=====================\n') ;

%%  loading

% Data originally come from: https://www.computationalimaging.org/publications/nlos-fk/
load('teaser.mat');
final_meas = final_meas(1:2:end,:,:) + final_meas(2:2:end,:,:);
final_meas = final_meas(:,1:2:end,:) + final_meas(:,2:2:end,:);
meas = final_meas;
meas = permute(meas,[3 2 1]).*mask;
%meas = permute(meas,[4 3 2 1]);
grid_z = single(repmat(linspace(0,1,bin)',[1 N N]));
data = single(zeros(bin,N,N,frame));

% FastDVDnet: Matias Tassano, Julie Delon, and Thomas Veit. CVPR, 2020, pp. 1354-1363
load('fastdvd.mat');

%%
for i = 1:frame
    data(:,:,:,i) = meas(:,:,:,i).*(grid_z.^2); 
end
data = gpuArray(data);
data = reshape(mtx*data(:,:),[bin N N]);

fprintf('=================2 loading done=====================\n') ;

%%
fpsf = ifftshift(fftn(fftshift(psf)));
blur = zeros(2*bin,2*N,2*N);
blur(2:2*bin,2:2*N,2:2*N) = flip(flip(flip(psf(2:2*bin,2:2*N,2:2*N),1),2),3);
fblur = ifftshift(fftn(fftshift(blur)));
fpsf = gpuArray(single(fpsf));
fblur = gpuArray(single(fblur));
clear blur psf
fprintf('=================start reconstruction=====================\n');


%%
fmin = 5;
fmax = 40;       % parameter for medium pass filtering
tvprameter = 0.00035;   % parameter for L1 regularization              
gate = 0;                 % not used           
maxiter = 10;   
tau = tvprameter*sum(data(:))/5;  


[result,~] = pnpstatic(N,bin,1,data,tau,fpsf,gate,mask,zeros(bin,N,N),maxiter,fblur,fmin,fmax); 
output = fastdvd(repmat(result,[1 1 1 5]),net,30,true,ones(128,128));
output = output(:,:,:,1);
picshow(output)
for loop = 1:3
    [result,~] = pnpstatic(N,bin,1,data,tau,fpsf,gate,mask,output,maxiter,fblur,fmin,fmax); 
    output = fastdvd(repmat(result,[1 1 1 5]),net,30,true,ones(128,128));
    output = output(:,:,:,1);
    picshow(output)
end


%% Used Function


function mask = definemask(N,samp,frame,bin)
    dd = N/samp; 
    xx = 2:dd:N;
    yy = 2:dd:N;
%     xx = N:-dd:1;
%     yy = N:-dd:1;
    mask = zeros(N,N); 
    for i = 1:samp
        for j = 1:samp
            mask(round(xx(i)),round(yy(j))) = 1;
        end
    end
    mask = reshape(mask,[1,N,N]);
    mask = repmat(mask,[bin 1 1 frame]);
end


function psf = definePsf(N,width,bin,timeRes,c)     % refer to lct reconstruction
    linexy = linspace(-2*width,2*width,2*N-1);            % linspace of scan range
    range = (bin*timeRes*c/2)^2;                    % total range of t^2 domain: bin^2*timeRes^2
    gridrange = bin*(timeRes*c/2)^2;                % one grid in t^2 domain: bin*timeRes^2
    [xx,yy,squarez] = meshgrid(linexy,linexy,0:gridrange:range);
    blur = abs((xx).^2+(yy).^2-squarez+0.000001);
    blur = double(blur == repmat(min(blur,[],3),[ 1 1 bin+1 ]));
    blur = blur(:,:,1:bin);                               % generate light-cone

    [x,y] = meshgrid(1:2*N-1,1:2*N-1);
    calib = max(N*1-sqrt((x-N-1).^2+(y-N-1).^2),0);
    calib = calib/max(calib(:))*2;
    %calib = exp(-((x-N-1).^2+(y-N-1).^2)/(N/3)^2);
    blur = blur.*calib;

    psf = zeros(2*N,2*N,2*bin);                       
    psf(2:2*N,2:2*N,bin+1:2*bin) = blur;                          % place it at center
    psf = permute(psf,[3 2 1]);  
end


function [mtx,mtxi] = resamplingOperator(M)   % refer to lct reconstruction
 % Local function that defines resampling operators
     mtx = sparse([],[],[],M.^2,M,M.^2);
     
     x = 1:M.^2;
     mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
     mtx  = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx*M/2;
     mtxi = mtx';
     
     K = log(M)./log(2);
     for k = 1:round(K)
          mtx  = 0.5.*(mtx(1:2:end,:)  + mtx(2:2:end,:));
          mtxi = 0.5.*(mtxi(:,1:2:end) + mtxi(:,2:2:end));
     end

     for i = 1:M
         mtxi(:,i) = mtxi(:,i)/sum(mtxi(:,i));
     end

end



