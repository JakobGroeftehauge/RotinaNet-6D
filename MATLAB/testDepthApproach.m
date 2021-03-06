clear; close all;
K = [   572.4114,   0.0,        325.2611;...
        0.0,        573.57043,  242.04899;...
        0.0,        0.0,        1.0];
gts = readmatrix('val_data_ape.csv');
x_bbox = mean([gts(:,2),gts(:,4)],2);
y_bbox = mean([gts(:,3),gts(:,5)],2);

x = gts(:,end-2);
y = gts(:,end-1);
z = gts(:,end);

x_calc = (x_bbox - K(1,3))/K(1,1).*z;
y_calc = (y_bbox - K(2,3))/K(2,2).*z;

x_diff = x_calc - x;
y_diff = y_calc - y;

figure
histogram(x_diff,50);
figure
histogram(y_diff,50);