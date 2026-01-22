%%
load HumanActivityTrain
load HumanActivityTest

%% Visualizando la secuencia de enternamiento
% X = XTrain{1}(1,:);
% 
% classes = categories(YTrain{1});
% 
% figure
% for j = 1:numel(classes)
%     label = classes(j);
%     idx = f 0.9997ind(YTrain{1} == label);
%     hold on
%     plot(idx,X(idx))
% end
% hold off
% 
% xlabel("Time Step")
% ylabel("Acceleration")
% title("Training Sequence 1, Feature 1")
% legend(classes,'Location','northwest')

%% Definiendo la arquitectura de la red neuronal
numFeatures = 3;
numHiddenUnits = 200;
numClasses = 5;

% ActivationFunctions = ["tanh_mex", "tanh_mex", "sigmoid_mex", "sigmoid_mex", "sigmoid_mex"];
% ActivationParameters = [0, 0, 0, 0 ,0];

k_param = @(a, b) 1 + exp(a .* b);
ActivationFunctions = ["zorro_tanh_mex", "zorro_tanh_mex", "zorro_sigm_mex", "sigmoid_mex", "zorro_sigm_mex"];
ActivationParameters = [...
    [1, 4, k_param(4, 2)], ...
    [1, 4, k_param(4, 2)], ...
    [0.5, 4, k_param(4, 2)], ...
    [0, 0, 0], ...
    [0.5, 4, k_param(4, 2)]];

layers = [ ...
    sequenceInputLayer(numFeatures)
    zlstmLayer(numHiddenUnits, ...
        'OutputMode','sequence', ... % set in order: A, Z, I, F, O
        'ActivationFunctions', ActivationFunctions, ...
        'ActivationParameters', ActivationParameters, ...
        'GeneralStrategy', 'ZLSTMStrategySimple')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Especificando las opciones para el entrenamiento
options = trainingOptions('adam', ...
    'MaxEpochs', 60, ... 
    'GradientThreshold',2, ...
    'Verbose',0, ...
    'ValidationData', {XTest{1}, YTest{1}}, ... 
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf, ...
    'Plots','training-progress' ...
);

%% Entrenando la red neuronal
t_init = tic();
[net, info] = trainNetwork(XTrain,YTrain,layers,options);
t_diff = toc(t_init);
disp(['Training time: ' num2str(t_diff) ' sec.'])

%% Cargando los datos para el testeo
figure
plot(XTest{1}')
xlabel("Time Step")
legend("Feature " + (1:numFeatures))
title("Test Data")

%% Realizando la clasificación en el conjunto de testeo
YPred = classify(net,XTest{1}, Acceleration="none");

%% Calculando el Acc en conjunto de testeo
acc = sum(YPred == YTest{1})./numel(YTest{1})

%% Comparando las predicciones en el conjunto d testeo
figure
plot(YPred,'.-')
hold on
plot(YTest{1})
hold off
xlabel("Time Step")
ylabel("Activity")
title("Predicted Activities")
legend(["Predicted" "Test Data"])

%save('humanActivity_net', "net")
%save('humanActivity_info', "info")