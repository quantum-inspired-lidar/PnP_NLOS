function threedshow(vol,range,width)


tic_z = linspace(0,range./2,size(vol,1));
tic_y = linspace(width,-width,size(vol,2));
tic_x = linspace(width,-width,size(vol,3));

% clip artifacts at boundary, rearrange for visualization
vol(end-10:end, :, :) = 0;
vol = permute(vol, [1, 2, 3]);

vol = flip(vol, 2);
vol = flip(vol, 3);

   % vol(tic_z < 0.9, :, :) = 0;
% View result
figure
% pic = squeeze(max(vol,[],1));
% pic = resizem(pic,[256,256]);
% imshow(pic,[]);
% % title('Front view');
% % set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
% % set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
% % xlabel('x (m)');
% % ylabel('y (m)');
% colormap('gray');
% axis square;

subplot(1,3,1);
imagesc(tic_x,tic_y,squeeze(max(vol,[],1)));
title('Front view');
set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
xlabel('x (m)');
ylabel('y (m)');
colormap('gray');
axis square;

subplot(1,3,2);
imagesc(tic_x,tic_z,squeeze(max(vol,[],2)));
title('Top view');
set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
set(gca,'YTick',linspace(min(tic_z),max(tic_z),3));
xlabel('x (m)');
ylabel('z (m)');
colormap('gray');
axis square;

subplot(1,3,3);
imagesc(tic_z,tic_y,squeeze(max(vol,[],3))')
title('Side view');
set(gca,'XTick',linspace(min(tic_z),max(tic_z),3));
set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
xlabel('z (m)');
ylabel('y (m)');
colormap('gray');
axis square;

drawnow;
%}
end