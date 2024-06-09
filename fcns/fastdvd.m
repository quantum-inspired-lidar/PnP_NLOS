function output = fastdvd(input,net,sigma,show,lamb)

    lamb = imresize(lamb,[256,256],"nearest");
    [bin,N,~,frame] = size(input);
    input = gather(input);
    zf = sign(input);
    input = abs(input);
    ref = zeros(N,N,frame);
    dep = zeros(N,N,frame);
    input(end-50:end, :, :,:) = 0;   
    for i = 1:frame
        % 3D to 2D ref and dep    
        [ref(:,:,i),dep(:,:,i)] = max(input(:,:,:,i),[],1);
        % nomalize
        netref(:,:,i) = imresize(ref(:,:,i),[256,256],'nearest');
        mr(i) = max(netref(:,:,i),[],'all');
        netref(:,:,i) = netref(:,:,i)/mr(i);
    end


    output = zeros(bin,N,N,frame);

   
  % FastDVDnet: Matias Tassano, Julie Delon, and Thomas Veit. CVPR, 2020, pp. 1354-1363  
    for i = 1:frame
        tempref = circshift(netref,-i+3,3);
        denoref(:,:,i) = predict(net,double(tempref),sigma/255.*lamb);
        newref(:,:,i) = imresize(denoref(:,:,i),[128,128],'nearest')*mr(i);
        % to 3D output

        for x = 1:N
            for y = 1:N
                if ref(x,y,i) == 0
                    output(dep(x,y),x,y,i) = newref(x,y,i);
                else
                    output(:,x,y,i) = newref(x,y,i)/ref(x,y,i)*input(:,x,y,i);
                end
            end
        end

    end

    output = zf.*output;

end
