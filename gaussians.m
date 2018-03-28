clear; clc;

mu1 = 2;
sigma_1 = 0.5;

mu2 = 2;
sigma_2 = 1.5;

mu_conv = (mu1 + mu2);
sigma_conv = sqrt(sigma_1^2 + sigma_2^2);

mu_mine = (mu1 + mu2)/2;
sigma_mine = sqrt((sigma_1^2 + sigma_2^2)/4);

x = linspace(min(-4*sigma_1 + mu1, -4*sigma_conv + mu_conv), ...
    max(4*sigma_2 + mu2, 4*sigma_conv + mu_conv),1000);

%% 1st gaussian like
y1a = 1/sqrt(2*pi*sigma_1^2);
y1b = exp(-(x-mu1).^2/(2*sigma_1^2));
y1 = y1a*y1b;

%% 2nd gaussian like
y2a = 1/sqrt(2*pi*sigma_2^2);
y2b = exp(-(x-mu2).^2/(2*sigma_2^2));
y2 = y2a * y2b;

%% Merging with convolution with precomputed values 
% http://www.tina-vision.net/docs/memos/2003-003.pdf
ya = 1/sqrt(2*pi*sigma_conv^2);
yb = exp(-(x-mu_conv).^2/(2*sigma_conv^2));
y = ya * yb;
hold on;

%% Mine Combination
% https://stats.stackexchange.com/questions/179213/mean-of-two-normal-distributions
y_minea = 1/sqrt(2*pi*sigma_mine^2);
y_mineb = exp(-(x-mu_mine).^2/(2*sigma_mine^2));
y_mine = y_minea * y_mineb;
hold on;

%% Plot

plot(x,y1);
hold on;
plot(x,y2);
hold on;
plot(x,y);
hold on;
plot(x,y_mine);
hold on;

legend('y_1','y_2','y_{convolution}','y_{mine}');
