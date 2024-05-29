function output = fastdvd(input,net,sigma,show,lamb)

    lamb = imresize(lamb,[256,256],"nearest");
    [bin,N,~,frame] = size(input);
    input = gather(input);
    zf = sign(input);
    input = abs(input);
    ref = zeros(N,N,frame);
    dep = zeros(N,N,frame);
    %input = convn(abs(input),ones(2,1,1,1)/2,'same');
    input(end-50:end, :, :,:) = 0;   
    for i = 1:frame
        % 3D to 2D ref and dep    
        [ref(:,:,i),dep(:,:,i)] = max(input(:,:,:,i),[],1);
        % nomalize
        netref(:,:,i) = imresize(ref(:,:,i),[256,256],'nearest');
        mr(i) = max(netref(:,:,i),[],'all');
        netref(:,:,i) = netref(:,:,i)/mr(i);
    end

%     for i = 1:frame
%         pad = zeros(148,148);
%         pad(11:138,11:138) = dep(:,:,i);
%         for x = 1:N
%             for y = 1:N
%                 if dep(x,y,i) == 1
%                     tj = pad(x:x+20,y:y+20);
%                     h = hist(tj(:),1:512);
%                     h(1) = 0;
%                     [~,dep(x,y,i)] = max(h);
%                 end
%             end
%         end
%     end

    output = zeros(bin,N,N,frame);

   
    % dvd denoising    
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

  
    %imshow(resizem(squeeze(max(output(:,:,:,1),[],1)),[256,256]),[]);

%     for i = 1:frame
%         if show
%             figure(5);
%             subplot(2,3,i);imshow(squeeze(max(output(:,:,:,i),[],1))',[]);
%             title([num2str(i)]);%colormap('hot');
%             set(gcf,'position',[100,150,1200,700]);  axis square; drawnow;
%         end
%     end

%    picshow(output(:,:,:,3));
    output = zf.*output;
%     blur = reshape([0.25,0.5,0.25],[3,1,1,1]);
%     output = convn(output,blur,'same');

end