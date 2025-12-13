clc; clear; close all;

data = readtable('C:\Users\qq293\OneDrive\桌面\Battery_dataset.csv');

batteries = unique(data.battery_id);

for i = 1:length(batteries)
    
    test_battery = batteries{i};
    
    train_data = data(~strcmp(data.battery_id, test_battery), :);
    test_data  = data(strcmp(data.battery_id, test_battery), :);
    
    X_train = train_data{:, {'cycle','chI','chV','chT','disI','disV','disT','BCt','SOH'}};
    y_train = train_data.RUL;
    X_test  = test_data{:, {'cycle','chI','chV','chT','disI','disV','disT','BCt','SOH'}};
    y_test  = test_data.RUL;
    
    mu = mean(X_train);
    sigma = std(X_train);
    X_train_norm = (X_train - mu) ./ sigma;
    X_test_norm  = (X_test - mu) ./ sigma;
    
   % train MLP
    hiddenLayerSize = [64 32];
    net = fitnet(hiddenLayerSize, 'trainlm'); 
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 1000;
   
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.0;

    net = train(net, X_train_norm', y_train');
    
    % Prediction
    y_pred = net(X_test_norm')';
    y_pred = max(0,y_pred);

    valid_idx = ~isnan(y_test) & ~isnan(y_pred) & (y_test ~= 0);
    y_true_valid = y_test(valid_idx);
    y_pred_valid = y_pred(valid_idx);
    
    MAPE = mean(abs((y_true_valid - y_pred_valid) ./ y_true_valid)) * 100;
    R2 = 1 - sum((y_true_valid - y_pred_valid).^2) / sum((y_true_valid - mean(y_true_valid)).^2);
    
    % plot
    figure;
    plot(test_data.cycle, y_test, 'b-', 'LineWidth', 2); hold on;
    plot(test_data.cycle(valid_idx), y_pred(valid_idx), 'r--', 'LineWidth', 2);
    grid on; xlabel('Cycle'); ylabel('RUL');
    legend('Real', 'Predicted', 'Location', 'best');
    title(sprintf('Case %d - Test Battery: %s\nMAPE=%.3f%%, R^2=%.3f', ...
        i, test_battery, MAPE, R2));
  
    saveas(gcf, sprintf('Case%d_%s_RUL_Prediction.png', i, test_battery));
    
end
