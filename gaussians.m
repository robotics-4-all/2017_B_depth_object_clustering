clear; clc;

mu1 = 2;
sigma_1 = 0.5;

mu2 = 4;
sigma_2 = 1.3;

mu_mine3 = (sigma_2^2 *mu1 + sigma_1^2 * mu2)/(sigma_1^2 + sigma_2^2);
sigma_mine3 = sqrt((sigma_1^2 + sigma_2^2)/2);

mu_mine = (mu1 + mu2)/2;
sigma_mine = sqrt((sigma_1^2 + sigma_2^2)/2);

mu_mine2 = mu_mine;
sigma_mine2 = sigma_mine;
for i = 1:20
    mu_mine2 = (mu1 + mu_mine2)/2;
    sigma_mine2 = sqrt((sigma_1^2 + sigma_mine2^2)/2);
end

x = linspace(min(-4*sigma_1 + mu1, -4*sigma_mine3 + mu_mine3), ...
    max(4*sigma_2 + mu2, 4*sigma_mine3 + mu_mine3),1000);

%% 1st gaussian like
y1a = 1/sqrt(2*pi*sigma_1^2);
y1b = exp(-(x-mu1).^2/(2*sigma_1^2));
y1 = y1a*y1b;

%% 2nd gaussian like
y2a = 1/sqrt(2*pi*sigma_2^2);
y2b = exp(-(x-mu2).^2/(2*sigma_2^2));
y2 = y2a * y2b;

%% Merging with mine3olution with precomputed values 
% http://www.tina-vision.net/docs/memos/2003-003.pdf
y_mine3a = 1/sqrt(2*pi*sigma_mine3^2);
y_mine3b = exp(-(x-mu_mine3).^2/(2*sigma_mine3^2));
y_mine3 = y_mine3a * y_mine3b;
hold on;

%% Mine Combination
% https://stats.stackexchange.com/questions/179213/mean-of-two-normal-distributions
y_minea = 1/sqrt(2*pi*sigma_mine^2);
y_mineb = exp(-(x-mu_mine).^2/(2*sigma_mine^2));
y_mine = y_minea * y_mineb;
hold on;

%% Mine Combination dummy
y_mine2a = 1/sqrt(2*pi*sigma_mine2^2);
y_mine2b = exp(-(x-mu_mine2).^2/(2*sigma_mine2^2));
y_mine2 = y_mine2a * y_mine2b;
hold on;

%% Plot

plot(x,y1);
hold on;
plot(x,y2);
hold on;
% plot(x,y);
% hold on;
% plot(x,y_mine);
% hold on;
plot(x,y_mine3);
hold on;

% legend('y_1','y_2','y_{mine3}','y_{mine}','^{y_{mine}+y_1}/_{2}');
legend('y_1','y_2','y_{mean}');

