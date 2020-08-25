clear; close all;

res = readmatrix('depth_preds.csv');

diff = abs(res(:,1)-res(:,2))*1000;

avg = mean(diff);

med = median(diff);

deviation = std(diff);

 