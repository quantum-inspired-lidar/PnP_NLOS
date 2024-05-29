function vol = FK(data,M,N,range,width)
tic
[z,y,x] = ndgrid(-M:M-1,-N:N-1,-N:N-1);
z = z./M; y = y./N; x = x./N;
grid_z = repmat(linspace(0,1,M)',[1 N N]);
data = data.*(grid_z.^1);
% Step 0: Pad data
data = sqrt(data);
tdata = zeros(2.*M,2.*N,2.*N);
tdata(1:end./2,1:end./2,1:end./2) = data;

% Step 1: FFT
tdata = fftshift(fftn(tdata));

% Step 2: Stolt trick
tvol = interpn(z,y,x,tdata,sqrt(abs((((N.*range)./(M.*width.*4)).^2).*(x.^2+y.^2)+z.^2)),y,x,'linear',0);
tvol = tvol.*(z > 0);
tvol = tvol.*abs(z)./max(sqrt(abs((((N.*range)./(M.*width.*4)).^2).*(x.^2+y.^2)+z.^2)),1e-6);

% Step 3: IFFT
tvol = ifftn(ifftshift(tvol));
tvol = abs(tvol).^2;   
vol = abs(tvol(1:end./2,1:end./2,1:end./2));

time_elapsed = toc;
display(sprintf(['Reconstructed volume of size %d x %d x %d '...
        'in %f seconds'], size(vol,3),size(vol,2),size(vol,1),time_elapsed));
end
