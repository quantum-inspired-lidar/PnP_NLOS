function picshow(x)
vol = gather(abs(x));    
vol = convn(vol,ones(2,1,1,1),'same');
vol(end-50:end, :, :) = 0;
%figure;
imshow(imresize(squeeze(max(vol,[],1)),[256*2,256*2],"nearest")',[],'border','tight');
end