% Script to create the 3d 3-part Gaussian Distribution.

N = 3.0;
sigma = 1;
mean = 2;
step = 0.005;
[X,Y] = meshgrid(-3 * sigma+mean:step:3 * sigma+mean);
R = (1/sqrt(2*pi*sigma*sigma).*exp(-((X-mean).^2/2)-((Y-mean).^2/2)));

% Coloring areas
C = X;
area_step = int16(sigma/step);

color1 = 0;
color2 = 1;
color3 = 2;

C(2*area_step:3*area_step,:) = color3;
C(end-3*area_step:end-2*area_step,:) = color3;
C(:,2*area_step:3*area_step) = color3;
C(:,end-3*area_step:end-2*area_step) = color3;

C(area_step:2*area_step,:) = color2;
C(end-2*area_step:end-area_step,:) = color2;
C(:,area_step:2*area_step) = color2;
C(:,end-2*area_step:end-area_step) = color2;

C(1:area_step,:) = color1;
C(end-area_step:end,:) = color1;
C(:,1:area_step) = color1;
C(:,end-area_step:end) = color1;


figure
mesh(X,Y,R,C)
xticks(-3 * sigma + mean:sigma:3 * sigma + mean)
yticks(-3 * sigma + mean:sigma:3 * sigma + mean)
yticklabels({'-3\sigma','-2\sigma','-\sigma','0','\sigma','2\sigma','3\sigma'})
xticklabels({'-3\sigma','-2\sigma','-\sigma','0','\sigma','2\sigma','3\sigma'})
xlim([-3*sigma+mean 3*sigma+mean]) 
ylim([-3*sigma+mean 3*sigma+mean])

title('3D Gaussian Distribution divided in 3 parts.')
xlabel('x axis') 
ylabel('y axis') 
zlabel('z axis')