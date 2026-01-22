%Load Sequence Data
data = chickenpox_dataset;
data = [data{:}];

numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

%Standardize Data
mu = mean(dataTrain);
sig = std(dataTrain);

% Prepare Predictors and Responses
dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%Forecast Future Time Steps
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
YTest = dataTestStandardized(2:end);

%Define LSTM Network Architecture
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

% ActivationFunctions = ["tanh_mex", "tanh_mex", "sigmoid_mex", "sigmoid_mex", "sigmoid_mex"];
% ActivationParameters = [0, 0, 0, 0 ,0];

k_param = @(a, b) 1 + exp(a .* b);
ActivationFunctions = ["zorro_tanh_mex", "zorro_tanh_mex", "zorro_sigm_mex", "zorro_sigm_mex", "zorro_sigm_mex"];
ActivationParameters = [...
    [1, 4, k_param(4, 2)], ...
    [1, 4, k_param(4, 2)], ...
    [0.5, 4, k_param(4, 2)], ...
    [0.5, 4, k_param(4, 2)], ...
    [0.5, 4, k_param(4, 2)]];

layers = [ ...
    sequenceInputLayer(numFeatures)
    zlstmLayer(numHiddenUnits, ... % set in order: A, Z, I, F, O
        'ActivationFunctions', ActivationFunctions, ...
        'ActivationParameters', ActivationParameters, ...
        'GeneralStrategy', 'ZLSTMStrategySimple')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions( ...
    'adam', ...
    'MaxEpochs',150, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',25, ...
    'LearnRateDropFactor',0.8, ...
    'ValidationData', {XTest, YTest}, ... 
    'ValidationFrequency',1, ...
    'ValidationPatience',Inf, ...
    'Plots','training-progress'); %'Verbose', 0, ...

%Train LSTM Network
t_init = tic();
[net, info] = trainNetwork(XTrain,YTrain,layers,options);
t_diff = toc(t_init);
disp(['Training time: ' num2str(t_diff) ' sec.'])

net = predictAndUpdateState(net,XTrain, Acceleration="none");
[net,YPred] = predictAndUpdateState(net,YTrain(end), Acceleration="none");

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),Acceleration="none");
end

YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse1 = sqrt(mean((YPred-YTest).^2))

figure(2); clf;
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])

figure(3); clf;
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
%title("RMSE = " + rmse)

%Update Network State with Observed Values
net = resetState(net);
net = predictAndUpdateState(net,XTrain, Acceleration="none");

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i), Acceleration="none");
end

YPred = sig*YPred + mu;

rmse1
rmse = sqrt(mean((YPred-YTest).^2))

figure(4); clf;
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)

%save('chickenPox_net', "net")
%save('chickenPox_info', "info")
