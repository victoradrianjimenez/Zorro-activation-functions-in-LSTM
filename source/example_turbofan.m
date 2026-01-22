dataFolder = '../data/CMAPSSData';

%% Prepare Training Data

filenamePredictors = fullfile(dataFolder,"train_FD001.txt");
[XTrain,YTrain] = processTurboFanDataTrain(filenamePredictors);

filenamePredictors = fullfile(dataFolder,"test_FD001.txt");
filenameResponses = fullfile(dataFolder,"RUL_FD001.txt");
[XTest,YTest] = processTurboFanDataTest(filenamePredictors,filenameResponses);

%% Remove Features with Constant Values

m = min([XTrain{:}],[],2);
M = max([XTrain{:}],[],2);
idxConstant = M == m;

for i = 1:numel(XTrain)
    XTrain{i}(idxConstant,:) = [];
end

numFeatures = size(XTrain{1},1)

%% Normalize Training Predictors

mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);
thr = 150;

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end
for i = 1:numel(XTest)
    XTest{i}(idxConstant,:) = [];
    XTest{i} = (XTest{i} - mu) ./ sig;
    YTest{i}(YTest{i} > thr) = thr;
end

%% Clip Responses

for i = 1:numel(YTrain)
    YTrain{i}(YTrain{i} > thr) = thr;
end

%% Prepare Data for Padding

for i=1:numel(XTrain)
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
XTrain = XTrain(idx);
YTrain = YTrain(idx);

% figure(1)
% bar(sequenceLengths)
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")

maxEpochs = 60; % 60
miniBatchSize = 20;

%% Define Network Architecture

numResponses = size(YTrain{1},1);
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
        'OutputMode','sequence', ...
        'ActivationFunctions', ActivationFunctions, ...
        'ActivationParameters', ActivationParameters, ...
        'GeneralStrategy', 'ZLSTMStrategySimple')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress', ...
    'ValidationData', {XTest, YTest}, ... 
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf ...
);

%% Train the Network

t_init = tic();
net = trainNetwork(XTrain,YTrain,layers,options);
t_diff = toc(t_init);

%% Test the Network

% YPred = predict(net,XTest,'MiniBatchSize',1);
net = resetState(net);
[net, YPred] = predictAndUpdateState(net, XTest, Acceleration="none");

idx = randperm(numel(YPred),4);
figure(2)
for i = 1:numel(idx)
    subplot(2,2,i)
    
    plot(YTest{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    ylim([0 thr + 25])
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("RUL")
end
legend(["Test Data" "Predicted"],'Location','southeast')

for i = 1:numel(YTest)
    YTestLast(i) = YTest{i}(end);
    YPredLast(i) = YPred{i}(end);
end
figure(3)
rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(YPredLast - YTestLast)
title("RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")



% figure(5); clf;
% plot(net.Layers(2).Trace(:,1:5))
% xlabel('Iterations');
% ylabel('Parameter value');
% legend('Activation','Z gate','Input','Forget','Output','location','best')
% 

disp(['Tiempo transcurrido: ' num2str(t_diff) ' seg.'])
