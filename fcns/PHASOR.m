function vol = PHASOR(data,M,N,fpsf,mtx,mtxi,wall_size,bin_resolution,sampling_coeff)
        tic
        bp_psf = conj(fpsf);
        grid_z = repmat(linspace(0,1,M)',[1 N N]);
        data = data.*(grid_z.^1.5);
        % Step 0: define virtual wavelet properties
        s_lamda_limit = wall_size/(N - 1); % sample spacing on the wall
        %sampling_coeff = 2; % scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2); % virtual wavelength in units of cm
        cycles = 5; % number of wave cycles in the wavelet, typically 4-6

        % Step 1: convolve measurement volume with virtual wave
        [phasor_data_cos, phasor_data_sin] = waveconv(bin_resolution, virtual_wavelength, cycles, data);
        phasor_data_cos = single(phasor_data_cos);
        phasor_data_sin = single(phasor_data_sin);
        
        % Step 2: transform virtual wavefield into LCT domain
        phasor_tdata_cos = single(zeros(2.*M,2.*N,2.*N));
        phasor_tdata_sin = single(zeros(2.*M,2.*N,2.*N));
        phasor_tdata_cos(1:end./2,1:end./2,1:end./2) = reshape(mtx*phasor_data_cos(:,:),[M N N]);
        phasor_tdata_sin(1:end./2,1:end./2,1:end./2) = reshape(mtx*phasor_data_sin(:,:),[M N N]);

        % Step 3: convolve with backprojection kernel
        tvol_phasorbp_sin = ifftn(fftn(phasor_tdata_sin).*bp_psf);
        tvol_phasorbp_sin = tvol_phasorbp_sin(1:end./2,1:end./2,1:end./2);
        phasor_tdata_cos = ifftn(fftn(phasor_tdata_cos).*bp_psf);       
        phasor_tdata_cos = phasor_tdata_cos(1:end./2,1:end./2,1:end./2);
        
        % Step 4: compute phasor field magnitude and inverse LCT
        tvol = sqrt(tvol_phasorbp_sin.^2 + phasor_tdata_cos.^2);
        %tvol = abs(phasor_tdata_cos);
        vol  = reshape(mtxi*tvol(:,:),[M N N]);
        vol  = max(real(vol),0);
        
        time_elapsed = toc;

    display(sprintf(['Reconstructed volume of size %d x %d x %d '...
        'in %f seconds'], size(vol,3),size(vol,2),size(vol,1),time_elapsed));
end