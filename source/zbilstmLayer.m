function layer = zbilstmLayer(numHiddenUnits, nameValueArgs)
%zbilstmLayer   Bidirectional Long Short-Term Memory (BiLSTM) layer
%
%   layer = zbilstmLayer(numHiddenUnits) creates a Bidirectional Long
%   Short-Term Memory layer. numHiddenUnits is the number of hidden units
%   in the forward and backward sequence LSTMs of the layer, specified as a
%   positive integer.
%
%   layer = zbilstmLayer(numHiddenUnits, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%     'Name'                              - Name for the layer, specified
%                                           as a character vector or a
%                                           string. The default value is
%                                           ''.
%     'InputWeights'                      - Input weights, specified by an
%                                           8*numHiddenUnits-by-D matrix or
%                                           [], where D is the number of
%                                           features of the input data. The
%                                           default is [].
%     'RecurrentWeights'                  - Recurrent weights, specified as 
%                                           an 8*numHiddenUnits-by-
%                                           numHiddenUnits
%                                           matrix or []. The default is
%                                           [].
%     'Bias'                              - Layer biases, specified as an 
%                                           8*numHiddenUnits-by-1 vector or
%                                           []. The default is [].
%     'HiddenState'                       - Initial hidden state, specified
%                                           as a 2*numHiddenUnits-by-1
%                                           vector or []. The default is
%                                           [].
%     'CellState'                         - Initial cell state, specified
%                                           as a 2*numHiddenUnits-by-1
%                                           vector or []. The default is
%                                           [].
%     'OutputMode'                        - The format of the output of the
%                                           layer. Options are:
%                                               - 'sequence', to output a
%                                               full sequence.
%                                               - 'last', to output the
%                                               last element only. 
%                                           The default value is
%                                           'sequence'.
%     'ActivationFunctions'               - Activation function for the 
%                                           hidden state and the gates: 
%                                           a, z, i, f, o
%                                           Options for hidden state and z:
%                                               - 'tanh'
%                                               - 'softsign'
%                                           The default value is 'tanh'.
%                                           Options for gates i, f and o:
%                                               - 'sigmoid'
%                                               - 'hard-sigmoid'
%                                           The default value is 'sigmoid'.
%     'ActivationParameters'              - Vector contains n parameters
%                                           per gate (5 x n in total).
%     'HasStateInputs'                    - Flag indicating whether the 
%                                           layer has extra inputs
%                                           that represent the layer
%                                           states. Specified as one of
%                                           the following:
%                                           - false - The layer has
%                                             a single input with the
%                                             name 'in'.
%                                           - true - The layer has
%                                             two additional inputs
%                                             with the names 'hidden'
%                                             and 'cell', which
%                                             correspond to the hidden
%                                             state and cell state,
%                                             respectively.
%                                           The default is false. When the
%                                           layer has state inputs,
%                                           HiddenState and CellState must
%                                           not be specified.
%     'HasStateOutputs'                   - Flag indicating whether the
%                                           layer should have extra
%                                           outputs that represent the
%                                           layer states. Specified as
%                                           one of the following:
%                                           - false - The layer has
%                                             a single output with the
%                                             name 'out'.
%                                           - true - The layer has
%                                             two additional outputs
%                                             with the names 'hidden'
%                                             and 'cell', which
%                                             correspond to the updated
%                                             hidden state and cell
%                                             state, respectively.
%                                           The default is false.
%     'InputWeightsLearnRateFactor'       - Multiplier for the learning 
%                                           rate of the input weights,
%                                           specified as a scalar or a
%                                           eight-element vector. The
%                                           default value is 1.
%     'RecurrentWeightsLearnRateFactor'   - Multiplier for the learning 
%                                           rate of the recurrent weights,
%                                           specified as a scalar or a
%                                           eight-element vector. The
%                                           default value is 1.
%     'BiasLearnRateFactor'               - Multiplier for the learning 
%                                           rate of the bias, specified as
%                                           a scalar or a eight-element
%                                           vector. The default value is 1.
%     'InputWeightsL2Factor'              - Multiplier for the L2
%                                           regularizer of the input
%                                           weights, specified as a scalar
%                                           or a eight-element vector. The
%                                           default value is 1.
%     'RecurrentWeightsL2Factor'          - Multiplier for the L2
%                                           regularizer of the recurrent
%                                           weights, specified as a scalar
%                                           or a eight-element vector. The
%                                           default value is 1.
%     'BiasL2Factor'                      - Multiplier for the L2
%                                           regularizer of the bias,
%                                           specified as a scalar or a
%                                           eight-element vector. The
%                                           default value is 0.
%     'InputWeightsInitializer'           - The function to initialize the
%                                           input weights, specified as
%                                           'glorot', 'he', 'orthogonal',
%                                           'narrow-normal', 'zeros',
%                                           'ones' or a function handle.
%                                           The default is 'glorot'.
%     'RecurrentWeightsInitializer'       - The function to initialize the
%                                           recurrent weights, specified as
%                                           'glorot', 'he', 'orthogonal',
%                                           'narrow-normal', 'zeros',
%                                           'ones' or a function handle.
%                                           The default is 'orthogonal'.
%     'BiasInitializer'                   - The function to initialize the
%                                           bias, specified as
%                                           'unit-forget-gate',
%                                           'narrow-normal', 'ones' or a
%                                           function handle. The default is
%                                           'unit-forget-gate'.
%
%   Example 1:
%       % Create a Bidirectional LSTM layer with 100 hidden units.
%
%       layer = zbilstmLayer(100);
%
%   Example 2:
%       % Create a Bidirectional LSTM layer with 50 hidden units which
%       % returns a single element. Manually initialize the recurrent weights
%       % from a Gaussian distribution with standard deviation 0.01
%
%       numHiddenUnits = 50;
%       layer = zbilstmLayer(numHiddenUnits, 'OutputMode', 'last', ...
%       'RecurrentWeights', randn([8*numHiddenUnits numHiddenUnits])*0.01);
%
%   See also nnet.cnn.layer.BiLSTMLayer
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2017-2021 The MathWorks, Inc.

% Parse the input arguments.
arguments
    numHiddenUnits (1,1) {mustBeNumeric, mustBeInteger, mustBePositive}
    nameValueArgs.Name  {mustBeText, iAssertValidLayerName} = ""
    nameValueArgs.OutputMode {mustBeText} = "sequence"
    nameValueArgs.ActivationFunctions = ["tanh" "tanh" "sigmoid" "sigmoid" "sigmoid"]
    nameValueArgs.ActivationParameters = []
    nameValueArgs.CustomLearnableParametersSize = []
    nameValueArgs.GeneralStrategy = 'ZLSTMGeneralStrategy'
    nameValueArgs.HasStateInputs (1,1) {iAssertBinary} = false
    nameValueArgs.HasStateOutputs (1,1) {iAssertBinary} = false
    nameValueArgs.InputWeights = []
    nameValueArgs.RecurrentWeights = []
    nameValueArgs.CustomLearnableParameters = []
    nameValueArgs.Bias = []
    nameValueArgs.InputWeightsLearnRateFactor (1,:) {iAssertValidFactor} = 1
    nameValueArgs.RecurrentWeightsLearnRateFactor (1,:) {iAssertValidFactor} = 1
    nameValueArgs.BiasLearnRateFactor (1,:) {iAssertValidFactor} = 1
    nameValueArgs.CustomLearnableParametersLearnRateFactor (1,:) {iAssertValidFactor} = 1
    nameValueArgs.InputWeightsL2Factor (1,:) {iAssertValidFactor} = 1
    nameValueArgs.RecurrentWeightsL2Factor (1,:) {iAssertValidFactor} = 1
    nameValueArgs.BiasL2Factor (1,:) {iAssertValidFactor} = 0
    nameValueArgs.CustomLearnableParametersL2Factor (1,:) {iAssertValidFactor} = 1
    nameValueArgs.InputWeightsInitializer = "glorot"
    nameValueArgs.RecurrentWeightsInitializer = "orthogonal"
    nameValueArgs.BiasInitializer = "unit-forget-gate"
    nameValueArgs.CustomLearnableParametersInitializer = "glorot"
    nameValueArgs.HiddenState = iDefaultState()
    nameValueArgs.CellState = iDefaultState()
end

% Canonicalization.
numHiddenUnits = double(gather(numHiddenUnits));
nameValueArgs = nnet.internal.cnn.layer.util.gatherParametersToCPU(nameValueArgs);
nameValueArgs.Name = convertStringsToChars(nameValueArgs.Name);
nameValueArgs.InputSize = [];
nameValueArgs.OutputMode = iAssertAndReturnValidOutputMode(nameValueArgs.OutputMode);
nameValueArgs.ActivationFunctions = nameValueArgs.ActivationFunctions; %%%%% DO NO CHECK
nameValueArgs.ActivationParameters = nameValueArgs.ActivationParameters; %%%%% DO NO CHECK
nameValueArgs.CustomLearnableParametersSize = nameValueArgs.CustomLearnableParametersSize; %%%%% DO NO CHECK
nameValueArgs.GeneralStrategy = nameValueArgs.GeneralStrategy; %%%%% DO NO CHECK
nameValueArgs.HasStateOutputs = logical(nameValueArgs.HasStateOutputs);
nameValueArgs.HiddenState = iAssertAndReturnValidState(nameValueArgs.HiddenState, nameValueArgs.HasStateInputs, "'HiddenState'");
nameValueArgs.CellState = iAssertAndReturnValidState(nameValueArgs.CellState, nameValueArgs.HasStateInputs, "'CellState'");

% Create an internal representation of the layer.
internalLayer = ZBiLSTMLayerInternal(nameValueArgs.Name, ... %%%%% nnet.internal.cnn.layer.BiLSTM
    nameValueArgs.InputSize, ...
    numHiddenUnits, ...
    true, ...
    true, ...
    iGetReturnSequence(nameValueArgs.OutputMode), ...
    nameValueArgs.ActivationFunctions, ...
    nameValueArgs.ActivationParameters, ...
    nameValueArgs.CustomLearnableParametersSize, ...
    nameValueArgs.GeneralStrategy, ...
    nameValueArgs.HasStateInputs, ...
    nameValueArgs.HasStateOutputs);

% Use the internal layer to construct a user visible layer.
layer = ZBiLSTMLayerExternal(internalLayer);

% Set learnable parameters, learn rate, L2 Factors and initializers.
layer.InputWeights = nameValueArgs.InputWeights;
layer.InputWeightsL2Factor = nameValueArgs.InputWeightsL2Factor;
layer.InputWeightsLearnRateFactor = nameValueArgs.InputWeightsLearnRateFactor;
layer.InputWeightsInitializer = nameValueArgs.InputWeightsInitializer;

layer.RecurrentWeights = nameValueArgs.RecurrentWeights;
layer.RecurrentWeightsL2Factor = nameValueArgs.RecurrentWeightsL2Factor;
layer.RecurrentWeightsLearnRateFactor = nameValueArgs.RecurrentWeightsLearnRateFactor;
layer.RecurrentWeightsInitializer = nameValueArgs.RecurrentWeightsInitializer;

layer.Bias = nameValueArgs.Bias;
layer.BiasL2Factor = nameValueArgs.BiasL2Factor;
layer.BiasLearnRateFactor = nameValueArgs.BiasLearnRateFactor;
layer.BiasInitializer = nameValueArgs.BiasInitializer;

layer.CustomLearnableParameters = nameValueArgs.CustomLearnableParameters;
layer.CustomLearnableParametersL2Factor = nameValueArgs.CustomLearnableParametersL2Factor;
layer.CustomLearnableParametersLearnRateFactor = nameValueArgs.CustomLearnableParametersLearnRateFactor;
% layer.CustomLearnableParametersInitializer = nameValueArgs.CustomLearnableParametersInitializer;

% Set hidden state and cell state.
if ~layer.HasStateInputs
    layer.HiddenState = nameValueArgs.HiddenState;
    layer.CellState = nameValueArgs.CellState;
end
end

function tf = iGetReturnSequence( mode )
tf = true;
if strcmp( mode, 'last' )
    tf = false;
end
end

function state = iDefaultState()
state = [];
end

function validString = iAssertAndReturnValidOutputMode(value)
validString = validatestring(value, {'sequence', 'last'});
end

function iAssertBinary(value)
nnet.internal.cnn.options.OptionsValidator.assertBinary(value);
end

function iAssertValidFactor(value)
validateattributes(value, {'numeric'},  {'vector', 'real', 'nonnegative', 'finite'});
end

function state = iAssertAndReturnValidState(state, hasStateInputs, name)
% If hasStateInputs is true, then only the default state is valid
if hasStateInputs && ~isequal(state, iDefaultState())
    error( message('nnet_cnn:layer:BiLSTMLayer:SettingStateWithStateInputs', name) );
end
end

function iAssertValidLayerName(name)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName(name));
end

function iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    func();
catch exception
    throwAsCaller(exception)
end
end