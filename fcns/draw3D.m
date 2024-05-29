function draw3D(data,width,thre,ma)
% thre = 0.2;
% width = 0.5;

data = flip(flip(data,2),3);
tic_x = linspace(-width,width,128);
tic_y = linspace(-width,width,128);
tic_Z = linspace(0,96e-4*512/2,512);
% tic_t = linspace(0, 32e-12*size(sig_in,3)/2, size(sig_in,3)/2);
[tic_X, tic_Y] = meshgrid(tic_x, tic_y);
[I,J] = size(tic_X);
tic_X = reshape(tic_X, [I*J 1]);
tic_Y = reshape(tic_Y, [I*J 1]);
[M,dep] = max(data, [], 1);
M = squeeze(M);
dep = squeeze(dep);
dep = tic_Z(dep);
M = reshape(M, [I*J 1]);
dep = reshape(dep, [I*J 1]);
M = M/max(M(:));
M = min(M,ma);
scatter3(dep(M > thre),tic_X(M > thre),tic_Y(M > thre),M(M > thre)*30,M(M > thre), 'o','filled','MarkerFaceAlpha',1);
axis([0 tic_Z(end) tic_x(1) tic_x(end) tic_y(1) tic_y(end)])
colormap('gray')
caxis([0 max(M(:))])
view([-64 19])

set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
set(gca,'ztick',[],'zticklabel',[])
% xlabel('z(mm)');
% ylabel('x(mm)');
% zlabel('y(mm)');
axis square;
set(gca,'color','k');
%set(1,'defaultfigurecolor', 'w');
set(gcf,'position',[700,350,600,600]) 
end
% [M,dep] = max(data, [], 1);
% mask = zeros(256,64,64);
% for i = 1:64
%     for j = 1:64
%         mask(dep(1,i,j),i,j) = 1;
%     end
% end
% data = data.*mask;
% data = zeros(256,64,64);
% for i  = 1:64
%     for j = 1:64
%         data(:,i,j) = squeeze(sum(sum(vol(:,4*i-3:4*i,4*j-3:4*j),2),3));
%     end
% end
