clear; close all;
preds = readmatrix('pre_file.csv');
preds = preds(:,2:end);
gts = readmatrix('val_data_ape.csv');
gts = gts(:,7:15);

for i=1:length(gts)
    clf;close all;
    pred = preds(i,:);
    gt = gts(i,:);
    
    matlab_ortho = reshape(pred,[3,3])';
    [U,S,V] = svd(matlab_ortho);
    d = det(V*U');
    matlab_ortho = V*[1,0,0;0,1,0;0,0,sign(d)]*U';
    ortho_pred = matlab_ortho;

    % raw prediction
    plot3([0,pred(1)],[0,pred(2)],[0,pred(3)],'r', 'LineWidth', 2)
    hold on
    plot3([0,pred(4)],[0,pred(5)],[0,pred(6)],'g', 'LineWidth', 2)
    hold on
    plot3([0,pred(7)],[0,pred(8)],[0,pred(9)],'b', 'LineWidth', 2)

    % reorthogonalized prediction
    plot3([0,ortho_pred(1)],[0,ortho_pred(2)],[0,ortho_pred(3)],'--r', 'LineWidth', 2)
    hold on
    plot3([0,ortho_pred(4)],[0,ortho_pred(5)],[0,ortho_pred(6)],'--g', 'LineWidth', 2)
    hold on
    plot3([0,ortho_pred(7)],[0,ortho_pred(8)],[0,ortho_pred(9)],'--b', 'LineWidth', 2)

    %ground truth
    plot3([0,gt(1)],[0,gt(2)],[0,gt(3)],'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2)
    hold on
    plot3([0,gt(4)],[0,gt(5)],[0,gt(6)],'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
    hold on
    plot3([0,gt(7)],[0,gt(8)],[0,gt(9)],'Color', [0 0.4470 0.7410], 'LineWidth', 2)

    pbaspect([1 1 1])
    grid on
    
    pause
end

pred = [0.13895652,1.0779688,0.10750908,0.63579506,0.027700096,-0.88151175,-0.91716504,0.20774172,-0.4023506];
ortho_pred = [0.1419254405078445,0.5022925545133632,-0.8529709155799221,0.9826282914933411,0.032603233205276444,0.18269828096290608,0.11957759004782043,-0.8640829026677335,-0.4889396260010148];
gt = [0.113309,0.99136102,0.0660649,0.58543199,-0.0128929,-0.810619,-0.802764,0.130527,-0.58183599,];

matlab_ortho = reshape(pred,[3,3])';
[U,S,V] = svd(matlab_ortho);
d = det(V*U');
matlab_ortho = V*[1,0,0;0,1,0;0,0,sign(d)]*U';
ortho_pred = matlab_ortho;

% raw prediction
plot3([0,pred(1)],[0,pred(2)],[0,pred(3)],'r', 'LineWidth', 2)
hold on
plot3([0,pred(4)],[0,pred(5)],[0,pred(6)],'g', 'LineWidth', 2)
hold on
plot3([0,pred(7)],[0,pred(8)],[0,pred(9)],'b', 'LineWidth', 2)

% reorthogonalized prediction
plot3([0,ortho_pred(1)],[0,ortho_pred(2)],[0,ortho_pred(3)],'--r', 'LineWidth', 2)
hold on
plot3([0,ortho_pred(4)],[0,ortho_pred(5)],[0,ortho_pred(6)],'--g', 'LineWidth', 2)
hold on
plot3([0,ortho_pred(7)],[0,ortho_pred(8)],[0,ortho_pred(9)],'--b', 'LineWidth', 2)

%ground truth
plot3([0,gt(1)],[0,gt(2)],[0,gt(3)],'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2)
hold on
plot3([0,gt(4)],[0,gt(5)],[0,gt(6)],'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2)
hold on
plot3([0,gt(7)],[0,gt(8)],[0,gt(9)],'Color', [0 0.4470 0.7410], 'LineWidth', 2)

pbaspect([1 1 1])
grid on