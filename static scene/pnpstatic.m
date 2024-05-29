function [x, iter] = pnpstatic(N, bin, frame,y, tau, fpsf, gate,mask, ...
    initialize,maxiter,fblur,fmin,fmax)

% Usage/Parameters: 
% pnpstatic(N, bin, frame,y, tau, fpsf, gate,mask, ...
%    initialize,maxiter,fblur,fmin,fmax)
%     N, bin, frame: size of the input data (bin x N x N x frame)
%     y: measurements 
%     tau: parameter for L1 regularization
%     fpsf, fblur: LCT convolutional kernel [Nature 555, 338 (2018)]
%     mask: downsampling operator
%     initialize: initial values for reconstruction
%     fmin, fmax: parameter for medium pass filtering



y = gpuArray(single(y));
fpsf = gpuArray(single(fpsf));
fblur = gpuArray(single(fblur));
mask = gpuArray(single(mask));

iter = 1;
 
alpha = 2^23*ones(1,frame);

miniter = 0;
alphamin = 1e-30;
alphamax = 1e15;

pada = gpuArray(single(zeros(2*bin,2*N,2*N)));
grad = gpuArray(single(zeros(bin,N,N,frame)));

x = gpuArray(single(initialize));

for i = 1:frame
    Ax(:,:,:,i) = forwardmodel(x(:,:,:,i),pada,fpsf,mask(:,:,:,i),bin,N); 
end

Axprevious = Ax;
xprevious = x;


%% = Begin Main Algorithm Loop =
% =============================
for i = 1:frame
iter = 1;
while (iter <= miniter) || ((iter <= maxiter))
    
        B = (y(:,:,:,i)-Ax(:,:,:,i))./(Ax(:,:,:,i)/10^4+1);       
        grad(:,:,:,i) = computegrad(B,pada,fblur,bin,N,'BP',fmin,fmax); 

        x(:,:,:,i) = xprevious(:,:,:,i) - grad(:,:,:,i)/alpha(1,i);
        x(:,:,:,i) = x(:,:,:,i)./(abs(x(:,:,:,i))+10^(-8)).*max( abs(x(:,:,:,i))- tau/alpha(1,i), 0 ); % 

        Ax(:,:,:,i) = forwardmodel(x(:,:,:,i),pada,fpsf,mask(:,:,:,i),bin,N); 
        normsqdx(1,i) = sum( (x(:,:,:,i) - xprevious(:,:,:,i)).^2 ,"all");

        gamma(1,i) = sum((Ax(:,:,:,i) - Axprevious(:,:,:,i)).^2,"all");
        if gamma(1,i) == 0
            alpha(1,i) = alphamin;
        else
            alpha(1,i) = gamma(1,i)./normsqdx(1,i);
            alpha(1,i) = min(alphamax, max(alpha(1,i), alphamin));
        end

        xprevious(:,:,:,i)  = x(:,:,:,i) ;
        Axprevious(:,:,:,i)  = Ax(:,:,:,i) ;     
        if mod(iter,5) == 0
        picshow(gather(abs(x(:,:,:,i))));
        iter
        end
        iter = iter + 1;
end

end
% ===========================
% = End Main Algorithm Loop =
iter = iter - 1;

end


function   grad = computegrad(B,pada,fblur,bin,N,method,fmin,fmax) 
    switch method 
        case 'BP'
            %B = y-Ax;
    
            fB = fft(B,512,1);
            fB(1:fmin,:,:,:) = 0;   %25 125
            fB(fmax:end,:,:,:) = 0;
            B = ifft(fB,512,1);

            pada(bin/2+1:bin*1.5,N/2:N*1.5-1,N/2:N*1.5-1) = B;
            fs = ifftshift(fftn(fftshift(pada)));  
            sb = ifftshift(ifftn(fftshift(fs.*fblur))); 
            grad = -real(sb(bin/2+1:bin*1.5,N/2:N*1.5-1,N/2:N*1.5-1)); 
    end
end



function   Ax = forwardmodel(x,pada,fpsf,mask,bin,N)

    pada(bin/2+1:bin*1.5,N/2:N*1.5-1,N/2:N*1.5-1) = x;
    fs = ifftshift(fftn(fftshift(pada)));  
    sb = ifftshift(ifftn(fftshift(fs.*fpsf))); 
    sb = real(sb(bin/2+1:bin*1.5,N/2:N*1.5-1,N/2:N*1.5-1));  %从结果中取出中间那块，大小等于small
    Ax= sb.*mask;

end


function [accuracy,deprmse] = depthevaluate(vol,scene)  
    for i = 1:128
        for j = 1:128
            [sceneref(i,j),scenedep(i,j)] =  max(squeeze(scene(:,i,j)));
        end
    end
    sceneref = sceneref/max(sceneref(:));
    indscene = sceneref>0.1;
    
    for i = 1:128
        for j = 1:128
            [front(i,j),dep(i,j)] =  max(squeeze(vol(:,i,j)));
        end
    end
    front = front/max(front(:));
    indfront = front>0.1;

    nullpix = (indscene+indfront == 0);
    exits = logical(indscene.*indfront);
    ind = logical( (abs(dep-scenedep)<10).*exits); % 
    accuracy = (sum(nullpix(:))+sum(ind(:)))/16384;
    numerd = round(sum(ind(:))*0.75);
    erd = sort( abs(dep(ind)-scenedep(ind)) );
    erd = erd(1:numerd);
    deprmse = sqrt(sum(erd.^2,'all')/numerd )*0.48; %cm
end
