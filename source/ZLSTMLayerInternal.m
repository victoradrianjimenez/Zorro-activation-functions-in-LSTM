classdef ZLSTMLayerInternal < nnet.internal.cnn.layer.FunctionalStatefulLayer ...
        & nnet.internal.cnn.layer.Recurrent...
        & nnet.internal.cnn.layer.CPUFusableLayer
    % LSTM   Implementation of the LSTM layer

    %   Copyright 2017-2021 The MathWorks, Inc.

    properties
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.learnable.PredictionLearnableParameter)
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % DynamicParameters   Dynamic parameters for the layer
        % (Vector of nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter)
        DynamicParameters = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter.empty();
        
        % InitialCellState   Initial value of the cell state
        InitialCellState
        
        % Initial hidden state   Initial value of the hidden state
        InitialHiddenState

        % Name (char array)   A name for the layer
        Name

        Trace
    end
    
    properties (Constant)
        % DefaultName   Default layer's name
        DefaultName = 'lstm'
    end
    
    properties (SetAccess = private)
        % InputSize (int) Size of the input vector
        InputSize
        
        % HiddenSize (int) Size of the hidden weights in the layer
        HiddenSize
        
        % ReturnSequence (logical) If true, output is a sequence. Otherwise
        % output is the last element in a sequence.
        ReturnSequence
        
        % RememberCellState (logical) If true, cell state is remembered
        RememberCellState
        
        % RememberHiddenState (logical) If true, hidden state is rememberedXTestNorm
        RememberHiddenState
        
        % Activation   (char) Activation function
        Activations

        % Custom constant parameters for activation functions
        ActParameters
        
        % HasStateInputs (logical) If true, layer has multiple inputs
        HasStateInputs
        
        % HasStateOutputs (logical) If true, layer has multiple outputs
        HasStateOutputs

        % flag that indicates if the activation parameters are learnables
        CustomLearnableParametersSize

        GeneralStrategy
    end
    
    properties (Access = private)
        ExecutionStrategy
    end
    
    properties (Dependent)
        % Learnable Parameters (nnet.internal.cnn.layer.LearnableParameter)
        InputWeights
        RecurrentWeights
        Bias
        CustomLearnableParameters  % custom learnable parameters for activation functions
        % Dynamic Parameters (nnet.internal.cnn.layer.DynamicParameter)
        CellState
        HiddenState
        % Learnables    Cell array of learnable parameters
        Learnables
        % State    Cell array with cell and hidden states
        State

        % for compatibility
        Activation
        RecurrentActivation
    end
    
    properties (SetAccess = protected, GetAccess={?nnet.internal.cnn.dlnetwork, ?nnet.internal.cnn.layer.FusedLayer})
        LearnablesNames = ["InputWeights" "RecurrentWeights" "Bias" "CustomLearnableParameters"]
    end    

    properties (SetAccess = protected)
        % IsInFunctionalMode   Returns true if layer is currently being
        % used in "functional" mode (i.e. in dlnetwork). Required by
        % FunctionalLayer interface. On construction, all layers are set up
        % for usage in DAGNetwork.
        IsInFunctionalMode = false

        % StateNames    Names of state parameters
        StateNames = ["HiddenState"; "CellState"]
    end
    
    properties (Dependent, SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   If the input size has not been determined, then this will be
        %   set to false, otherwise it will be true
        HasSizeDetermined
        
        % InputNames   Input names of the layer
        InputNames
        
        % OutputNames   Output names of the layer
        OutputNames
        
        % NumStates   Number of layer states
        NumStates
    end
    
    properties (Constant, Access = private)
        % InputWeightsIndex   Index of the Weights into the
        % LearnableParameters vector
        InputWeightsIndex = 1;
        
        % RecurentWeightsIndex   Index of the Recurrent Weights into the
        % LearnableParameters vector
        RecurrentWeightsIndex = 2;
        
        % BiasIndex   Index of the Bias into the LearnableParameters vector
        BiasIndex = 3;
        
        % CustomLearnableParametersIndex   Index of the Zorro into the LearnableParameters vector
        CustomLearnableParametersIndex = 4;

        % HiddenStateIndex   Index of the hidden state into the
        % DynamicParameters vector
        HiddenStateIndex = 1;
        
        % CellStateIndex   Index of the cell state into the
        % DynamicParameters vector
        CellStateIndex = 2;
    end
    
    methods
        function this = ZLSTMLayerInternal(name, inputSize, hiddenSize, ...
                rememberCellState, rememberHiddenState, returnSequence, ...
                activations, actParameters, customLearnableParametersSize, ...
                generalStrategy, hasStateInputs, hasStateOutputs, learnable_initializer)
            % ZLSTM   Constructor for an LSTM layer
            %
            %   Create an LSTM layer with the following
            %   compulsory parameters:
            %
            %   name                - Name for the layer [char array]
            %   inputSize           - Size of the input vector [int]
            %   hiddenSize          - Size of the hidden units [int]
            %   rememberCellState   - Remember the cell state [logical]
            %   rememberHiddenState - Remember the hidden state [logical]
            %   returnSequence      - Output as a sequence [logical]
            %   activations         - Activation function for hidden state and Z, I, F, O gates 
            %   actParameters       - parameters for custom activation function. Use [] to set it as learnables
            %   hasStateInputs      - Require states as inputs [logical]
            %   hasStateOutputs     - Return states as outputs [logical]
           
            % Set layer name
            this.Name = name;
            
            % Set parameters
            this.InputSize = inputSize;
            this.HiddenSize = hiddenSize;
            this.RememberCellState = rememberCellState;
            this.RememberHiddenState = rememberHiddenState;
            this.ReturnSequence = returnSequence;
            this.Activations = activations;
            if nargin < 8
                actParameters = [];
            end
            if nargin < 9
                customLearnableParametersSize = [];
            end
            if nargin < 10
                generalStrategy = 'ZLSTMGeneralStrategy';
            end
            if nargin < 11
                hasStateInputs = false;
                hasStateOutputs = false;
            end
            if nargin < 13
                learnable_initializer = 'zeros';
            end
            this.GeneralStrategy = generalStrategy;
            this.CustomLearnableParametersSize = customLearnableParametersSize;
            this.HasStateInputs = hasStateInputs;
            this.HasStateOutputs = hasStateOutputs;
            this.Trace = [];

            % Set weights and bias to be LearnableParameter
            this.InputWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.RecurrentWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.CustomLearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.ActParameters = reshape(actParameters',[],1);

            % Set default initializers. The external layer constructor
            % overwrites these values, which are set only for internal code
            % that bypasses the casual API.
            % options: "narrow-normal", "glorot", "he", "orthogonal", "ones", "zeros", "unit-forget-gate"
            this.InputWeights.Initializer = iInternalInitializer('narrow-normal');
            this.RecurrentWeights.Initializer = iInternalInitializer('narrow-normal');
            this.Bias.Initializer = iInternalInitializer('zeros');
            this.CustomLearnableParameters.Initializer = iInternalInitializer(learnable_initializer);
            
            % Set dynamic parameters
            this.CellState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
            this.HiddenState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter();
            
            % Initialize with host execution strategy
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function [Z, memory] = forward( this, X )
            % forward   Forward input data through the layer at training
            % time and output the result
            [Z, memory] = this.ExecutionStrategy.forward( ...
                X, this.InputWeights.Value, this.RecurrentWeights.Value, ...
                this.Bias.Value, this.CustomLearnableParameters.Value, ...
                this.CellState.Value, this.HiddenState.Value);

            if this.IsInFunctionalMode
                memory = this.computeFunctionalState(Z, memory);
            end

        end
        
        function [Z, state] = predict( this, X )
            % predict   Forward input data through the layer at prediction
            % time and output the result
            if nargout == 2
                [Z, state] = forward(this, X);
            else
                Z = forward(this, X);
            end
        end
        
        function [dX, dW] = backward( this, X, Z, dZ, memory, ~ )
            % backward    Back propagate the derivative of the loss function
            % through the layer
            needsWeightGradients = nargout > 1;
            if ~needsWeightGradients
                dX = this.ExecutionStrategy.backward(...
                    X, this.InputWeights.Value, this.RecurrentWeights.Value, ...
                    this.Bias.Value, this.CustomLearnableParameters.Value, ...
                    this.CellState.Value, this.HiddenState.Value, ...
                    Z, memory, dZ);
            else
                [dX, dIW, dRW, dB, dAP] = this.ExecutionStrategy.backward(...
                    X, this.InputWeights.Value, this.RecurrentWeights.Value, ...
                    this.Bias.Value, this.CustomLearnableParameters.Value, ...
                    this.CellState.Value, this.HiddenState.Value, ...
                    Z, memory, dZ);
                dW{this.InputWeightsIndex} = dIW;
                dW{this.RecurrentWeightsIndex} = dRW;
                dW{this.BiasIndex} = dB;
                dW{this.CustomLearnableParametersIndex} = dAP;
            end
        end
        
        function Zs = forwardExampleInputs(this, Xs)
            % Get input placeholder array
            X = Xs{1};
            
            % Validate input size and formats
            featureSize = getSizeForDims(X,'SC');
            this.assertInputIsScalarInDAGNetwork(featureSize)
            this.assertIsConsistentWithInferredSize(featureSize)
            if this.HasStateInputs
                this.assertValidStateInput(Xs{2});
                this.assertValidStateInput(Xs{3});
            end
            
            % The output of LSTM has no spatial dimensions, the number of
            % output channels is the hidden size, and the sequence length
            % either stays unchanged (OutputMode='sequence') or is removed.
            Z = setSizeForDim(X,'S',[]);
            Z = setSizeForDim(Z,'C',this.HiddenSize);
            if ~this.ReturnSequence
                Z = setSizeForDim(Z,'T',[]);
            end
            
            % If there is only one labelled output dimension, label the
            % second with 'U'
            if ndims(Z) == 1
                Z = setSizeForDim(Z,'U',1);
            end
            
            % Assign output placeholder arrays. If the layer has state
            % outputs, the state output dimensions are the same as the
            % layer output, but with no time dimension
            if this.HasStateOutputs
                state = setSizeForDim(Z, 'T', []);
                Zs = {Z, state, state};
            else
                Zs = {Z};
            end
        end
            
        function this = configureForInputs(this,Xs)
            % Get input placeholder array
            X = Xs{1};
            
            % Validate input size and formats
            featureSize = getSizeForDims(X,'SC');
            this.assertInputIsScalarInDAGNetwork(featureSize)
            this.assertIsConsistentWithInferredSize(featureSize)
            if this.HasStateInputs
                this.assertValidStateInput(Xs{2});
                this.assertValidStateInput(Xs{3});
            end
            
            % Store the number of input features in the layer
            if ~this.HasSizeDetermined
                this.InputSize = prod(featureSize);
            end
        end
        
        function out = forwardPropagateSequenceLength(~, ~, ~)
            out = {};
            error("Temporary internal error: forwardPropagateSequenceLength "+...
                "should not be called on an LSTM layer anymore")
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters    Initialize learnable
            % parameters using their initializer
            
            if isempty(this.InputWeights.Value)
                % Initialize only if it is empty
                weightsSize = [4*this.HiddenSize, this.InputSize];
                weights = this.InputWeights.Initializer.initialize(...
                    weightsSize, 'InputWeights');
                this.InputWeights.Value = precision.cast(weights);
            else
                % Cast to desired precision
                this.InputWeights.Value = precision.cast(this.InputWeights.Value);
            end
            
            if isempty(this.RecurrentWeights.Value)
                % Initialize only if it is empty
                weightsSize = [4*this.HiddenSize, this.HiddenSize];
                weights = this.RecurrentWeights.Initializer.initialize(...
                    weightsSize, 'RecurrentWeights');
                this.RecurrentWeights.Value = precision.cast(weights);
            else
                % Cast to desired precision
                this.RecurrentWeights.Value = precision.cast(this.RecurrentWeights.Value);
            end
            
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                biasSize = [4*this.HiddenSize, 1];               
                bias = this.Bias.Initializer.initialize(biasSize, 'Bias');
                this.Bias.Value = precision.cast(bias);
            else
                % Cast to desired precision
                this.Bias.Value = precision.cast(this.Bias.Value);
            end
            
            if ~isempty(this.CustomLearnableParametersSize)
                if isempty(this.CustomLearnableParameters.Value)
                     customLearnableParameters = this.CustomLearnableParameters.Initializer.initialize(...
                         this.CustomLearnableParametersSize, 'CustomLearnableParameters');
                else
                    customLearnableParameters = this.CustomLearnableParameters.Value;
                end
                this.CustomLearnableParameters.Value = precision.cast(customLearnableParameters);
            end
        end
        
        function this = initializeDynamicParameters(this, precision)
           % initializeDynamicParameters   Initialize dynamic parameters
           
           % Cell state
           if isempty(this.InitialCellState)
               parameterSize = [this.HiddenSize 1];
               this.InitialCellState = iInitializeConstant(parameterSize, precision);
           else
               this.InitialCellState = precision.cast(this.InitialCellState);
           end
           % Set the running cell state
           this.CellState.Value = this.InitialCellState;
           this.CellState.Remember = this.RememberCellState;
           
           % Hidden units
           if isempty(this.InitialHiddenState)
               parameterSize = [this.HiddenSize 1];
               this.InitialHiddenState = iInitializeConstant(parameterSize, precision);
           else
               this.InitialHiddenState = precision.cast(this.InitialHiddenState);
           end
           % Set the running hidden state
           this.HiddenState.Value = this.InitialHiddenState;
           this.HiddenState.Remember = this.RememberHiddenState;
           
           if this.IsInFunctionalMode
             this.CellState.Value = dlarray(this.CellState.Value);
             this.HiddenState.Value = dlarray(this.HiddenState.Value);
           end
        end
        
        function state = computeState(this, ~, Z, memory, ~)
            % state{1} - store hidden state
            % state{2} - store cell state
            state = cell(1,2);
            if ~this.HasStateInputs
                if this.HasStateOutputs
                    state = Z(2:3);
                else
                    state = {Z(:, :, end) memory.CellState(:, :, end)};
                end
            end
        end
        
        function this = updateState(this, state)

            if numel(this.CustomLearnableParametersSize) > 0
                this.Trace = [this.Trace; reshape(this.CustomLearnableParameters.Value, 1, [])];
            end

            if ~this.HasStateInputs
                % Update the hidden state
                this.DynamicParameters(this.HiddenStateIndex).Value = state{1};
                
                % Update the cell state
                this.DynamicParameters(this.CellStateIndex).Value = state{2};
            end
        end   
        
        function this = resetState(this)
            if ~this.HasStateInputs
                hiddenState = this.InitialHiddenState;
                cellState = this.InitialCellState;
                if this.IsInFunctionalMode
                    hiddenState = dlarray( hiddenState );
                    cellState = dlarray( cellState );
                end
                % Set the hidden state
                this.DynamicParameters(this.HiddenStateIndex).Value = hiddenState;

                % Set the cell state
                this.DynamicParameters(this.CellStateIndex).Value = cellState;
            end
        end
        
        
        function this = prepareForTraining(this)
            % prepareForTraining   Prepare this layer for training
            %   Before this layer can be used for training, we need to
            %   convert the learnable parameters to use the class
            %   TrainingLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = this.getHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
            this.LearnableParameters(3).UseGPU = false;
            if ~isempty(this.CustomLearnableParametersSize)
                this.LearnableParameters(4).UseGPU = false;
            end
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = this.getGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
            this.LearnableParameters(3).UseGPU = true;
            if ~isempty(this.CustomLearnableParametersSize)
                this.LearnableParameters(4).UseGPU = true;
            end
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = this.getGPUStrategy();
        end
        
        % Setter and getter for InputWeights, RecurrentWeights and Bias
        % These make easier to address into the vector of LearnableParameters
        % giving a name to each index of the vector
        function weights = get.InputWeights(this)
            weights = this.LearnableParameters(this.InputWeightsIndex);
        end
        
        function this = set.InputWeights(this, weights)
            this.LearnableParameters(this.InputWeightsIndex) = weights;
        end
        
        function weights = get.RecurrentWeights(this)
            weights = this.LearnableParameters(this.RecurrentWeightsIndex);
        end
        
        function this = set.RecurrentWeights(this, weights)
            this.LearnableParameters(this.RecurrentWeightsIndex) = weights;
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            this.LearnableParameters(this.BiasIndex) = bias;
        end
                
        function customLearnableParameters = get.CustomLearnableParameters(this)
            customLearnableParameters = this.LearnableParameters(this.CustomLearnableParametersIndex);
        end
        
        function this = set.CustomLearnableParameters(this, customLearnableParameters)
            this.LearnableParameters(this.CustomLearnableParametersIndex) = customLearnableParameters;
        end

        % Setter and getter for CellState and HiddenState
        function state = get.CellState(this)
            state = this.DynamicParameters(this.CellStateIndex);
        end
        
        function this = set.CellState(this, state)
            this.DynamicParameters(this.CellStateIndex) = state;
        end
        
        function state = get.HiddenState(this)
            state = this.DynamicParameters(this.HiddenStateIndex);
        end
        
        function this = set.HiddenState(this, state)
            this.DynamicParameters(this.HiddenStateIndex) = state;
        end
        
        % Getter for HasSizeDetermined
        function tf = get.HasSizeDetermined(this)
            tf = ~isempty( this.InputSize );
        end 
        
        function learnables = get.Learnables(this)
            % Assume setupForFunctional has been called
            inW = this.InputWeights.Value;
            recW = this.RecurrentWeights.Value;
            b = this.Bias.Value;
            learnables = {inW, recW, b};
            if ~isempty(this.CustomLearnableParametersSize)
                learnables = [learnables, this.CustomLearnableParameters.Value];
            end
        end
        
        function this = set.Learnables(this, learnables)
            % Assume setupForFunctional has been called
            hiddenSize = this.HiddenSize;
            nnet.internal.cnn.layer.paramvalidation.assertValidLearnables(learnables{1}, [4*hiddenSize, this.InputSize]);
            nnet.internal.cnn.layer.paramvalidation.assertValidLearnables(learnables{2}, [4*hiddenSize, hiddenSize]);
            nnet.internal.cnn.layer.paramvalidation.assertValidLearnables(learnables{3}, [4*hiddenSize, 1]);
            if ~isempty(this.CustomLearnableParametersSize)
                nnet.internal.cnn.layer.paramvalidation.assertValidLearnables(learnables{4}, this.CustomLearnableParametersSize);
            end
            this.LearnableParameters(this.InputWeightsIndex).Value = learnables{1};
            this.LearnableParameters(this.RecurrentWeightsIndex).Value = learnables{2};
            this.LearnableParameters(this.BiasIndex).Value = learnables{3};
            if ~isempty(this.CustomLearnableParametersSize)
                this.LearnableParameters(this.CustomLearnableParametersIndex).Value = learnables{4};
            end
        end

        function state = get.State(this)
            if this.HasStateInputs
                state = nnet.internal.cnn.layer.util.ParameterMarker.create(2);
            else
                state = {this.HiddenState.Value, this.CellState.Value};
            end
        end
        
        function this = set.State(this, state)
            marker = nnet.internal.cnn.layer.util.ParameterMarker.isMarker(state);
            if this.HasStateInputs && ~all(marker)
                error(message('nnet_cnn:internal:cnn:layer:LSTM:SettingStateWithStateInputs'));
            end

            hiddenState = state{1};
            cellState = state{2};

            expectedDataClass = {'single'};
            expectedClass = [expectedDataClass, {'dlarray'}];
            expectedAttributes = {'size', [this.HiddenSize NaN]};
            if ~marker(1)
                validateattributes(cellState, expectedClass, expectedAttributes);
                if isdlarray(cellState)
                    nnet.internal.cnn.layer.paramvalidation.validateStateDlarray(cellState, expectedDataClass, this.StateNames{1});
                elseif this.IsInFunctionalMode
                    cellState = dlarray(cellState);
                end
                this.CellState.Value = cellState;
            end
            if ~marker(2)
                validateattributes(hiddenState, expectedClass, expectedAttributes);
                if isdlarray(hiddenState)
                    nnet.internal.cnn.layer.paramvalidation.validateStateDlarray(hiddenState, expectedDataClass, this.StateNames{2});
                elseif this.IsInFunctionalMode
                    hiddenState = dlarray(hiddenState);
                end
                this.HiddenState.Value = hiddenState;
            end
        end
        
        function numStates = get.NumStates(this)
            if this.HasStateInputs
                numStates = 0;
            else
                numStates = 2;
            end
        end
        
        function names = get.InputNames(this)
            if this.HasStateInputs
                names = {'in', 'hidden', 'cell'};
            else
                names = {'in'};
            end
        end
        
        function names = get.OutputNames(this)
            if this.HasStateOutputs
                names = {'out', 'hidden', 'cell'};
            else
                names = {'out'};
            end
        end

        function activation = get.Activation(this)
            n = numel(this.ActParameters)/5;
            k = 5;
            activation = strcat(this.Activations(k), num2str(this.ActParameters(1+n*(k-1):n*k)', '_%d'));
        end

        function activations = get.RecurrentActivation(this)
            activations = strcat(strjoin(this.Activations,'_'), num2str(this.ActParameters(:)', '_%d'));
        end
    end
    
    methods (Access = private)
        function strategy = getHostStrategy(this)
            if needsGeneralStrategy(this)
                strategy = this.getGeneralStrategy();
            elseif this.ReturnSequence
                strategy = nnet.internal.cnn.layer.util.LSTMHostStrategy();
            else
                strategy = nnet.internal.cnn.layer.util.LSTMHostReturnLastStrategy();
            end
        end
        
        function strategy = getGPUStrategy(this)
            if needsGeneralStrategy(this)
                strategy = this.getGeneralStrategy();
            elseif this.ReturnSequence
                strategy = nnet.internal.cnn.layer.util.LSTMGPUStrategy();
            else
                strategy = nnet.internal.cnn.layer.util.LSTMGPUReturnLastStrategy();
            end
        end
        
        function strategy = getGeneralStrategy(this)
            strategy = feval(this.GeneralStrategy, ...
                this.Activations, ...
                this.ActParameters, ...
                this.ReturnSequence );
            
            if this.HasStateInputs
                strategy = nnet.internal.cnn.layer.util.LSTMMultiInputDecorator(strategy);
            end
            
            if this.HasStateOutputs
                strategy = nnet.internal.cnn.layer.util.LSTMMultiOutputDecorator(strategy);
            end
        end
        
        function tf = needsGeneralStrategy(this)
            % The layer does not need the general strategy whenever it has
            % the default activations and single-inputs, single-outputs.
            tf = (this.HasStateInputs || this.HasStateOutputs) || ~this.hasDefaultActivations();
        end
        
        function tf = hasDefaultActivations(this)
            %tf = ismember( this.Activation, {'tanh'} ) && ismember( this.RecurrentActivation, {'sigmoid'} );
            tf = 0;
        end
        
        function memory = computeFunctionalState(this, Z, memory)
            if this.HasStateInputs
                memory = nnet.internal.cnn.layer.util.ParameterMarker.create(2);
            elseif this.HasStateOutputs
                memory = {stripdims(Z{2}), stripdims(Z{3})};
            else
                memory = {memory.HiddenState, memory.CellState};
            end
        end

        function assertInputIsScalarInDAGNetwork(this,featureSize)
            if ~this.IsInFunctionalMode && ~isscalar(featureSize)
                iThrowNonScalarInputSizeError(featureSize);
            end
        end
        
        function assertIsConsistentWithInferredSize(this,featureSize)
            assert(~this.HasSizeDetermined || isequal(this.InputSize, prod(featureSize)) )
        end
        
        function assertValidStateInput(this, Xs)
            fmt = dims(Xs);
            if contains(fmt, 'S') || contains(fmt, 'T') || contains(fmt, 'U')
                iThrowStateInputMustBeCB();
            end
            stateSize = getSizeForDims(Xs, 'C');
            if ~isequal(stateSize, this.HiddenSize)
                iThrowStateInputMustMatchHiddenSize();
            end
        end
    end
    
    methods (Access = protected)
        function this = setFunctionalStrategy(this)
            %if ~this.hasDefaultActivations()
            %    error(message('nnet_cnn:internal:cnn:layer:LSTM:NonDefaultActivFunctional'))
            %end
            
            if this.ReturnSequence
                strategy = nnet.internal.cnn.layer.util.LSTMFunctionalStrategy;
            else
                strategy = nnet.internal.cnn.layer.util.LSTMFunctionalReturnLastStrategy;
            end
            
            if this.HasStateInputs
                strategy = nnet.internal.cnn.layer.util.LSTMMultiInputFunctionalDecorator(strategy);
            end
            
            if this.HasStateOutputs
                strategy = nnet.internal.cnn.layer.util.LSTMMultiOutputFunctionalDecorator(strategy);
            end
            
            this.ExecutionStrategy = strategy;
        end

        function this = initializeStates(this)
            % initializeStates    Zero as init value if empty
            precision = nnet.internal.cnn.util.Precision('single');
            this = initializeDynamicParameters(this, precision);
        end        
    end
    
    methods (Hidden)
        function layerArgs = getFusedArguments(layer)
            % getFusedArguments  Returned the arguments needed to call the
            % layer in a fused network.
            useGeneral = isa(layer.getHostStrategy(), layer.GeneralStrategy);
            layerArgs = { 'zlstm', layer.InputWeights.Value, ...
                layer.RecurrentWeights.Value, layer.Bias.Value, ...
                layer.CellState.Value, layer.HiddenState.Value, ...
                layer.Activations, useGeneral, layer.ReturnSequence };
        end

        function tf = isFusable(this)
            % isFusable  Indicates if the layer is fusable in a given network.
            tf = true;
            
            % Fusion is not enabled when the layer has state inputs or
            % outputs
            if this.HasStateInputs || this.HasStateOutputs
                tf = false;
            end
        end
    end
end

function parameter = iInitializeConstant(parameterSize, precision)
parameter = precision.cast( zeros(parameterSize) );
end

function initializer = iInternalInitializer(name)
initializer = nnet.internal.cnn.layer.learnable.initializer.initializerFactory(name);
end

function str = iSizeToString(sz)
str = join(string(sz), matlab.internal.display.getDimensionSpecifier);
end

function iThrowNonScalarInputSizeError(inputSize)
error( message('nnet_cnn:internal:cnn:layer:LSTM:NonScalarInputSize', iSizeToString(inputSize)) );
end

function iThrowStateInputMustBeCB()
error( message('nnet_cnn:internal:cnn:layer:LSTM:StateInputMustBeCB') );
end

function iThrowStateInputMustMatchHiddenSize()
error( message('nnet_cnn:internal:cnn:layer:LSTM:StateInputMustMatchHiddenSize') );
end
