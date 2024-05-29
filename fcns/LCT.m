function vol = LCT(data,M,N,fpsf,mtx,mtxi,snr)
tic;
grid_z = repmat(linspace(0,1,M)',[1 N N]);
data = data.*(grid_z.^3);
invpsf = conj(fpsf) ./ (abs(fpsf).^2 + 1./snr);

% Step 2: Resample time axis and pad result
tdata = zeros(2.*M,2.*N,2.*N);
tdata(1:end./2,1:end./2,1:end./2)  = reshape(mtx*data(:,:),[M N N]);

% Step 3: Convolve with inverse filter and unpad result
tvol = ifftn(fftn(tdata).*invpsf);
tvol = tvol(1:end./2,1:end./2,1:end./2);

% Step 4: Resample depth axis and clamp results
vol  = reshape(mtxi*tvol(:,:),[M N N]);
vol  = max(real(vol),0);
time_elapsed = toc;

display(sprintf(['Reconstructed volume of size %d x %d x %d '...
    'in %f seconds'], size(vol,3),size(vol,2),size(vol,1),time_elapsed));
end