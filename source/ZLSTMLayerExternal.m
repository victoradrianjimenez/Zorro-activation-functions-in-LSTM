classdef ZLSTMLayerExternal < nnet.cnn.layer.LSTMLayer

    properties (Dependent)
        CustomLearnableParameters
        CustomLearnableParametersLearnRateFactor (1,:) {mustBeNumeric}
        CustomLearnableParametersL2Factor (1,:) {mustBeNumeric}
        Trace
    end

    methods
        function val = get.CustomLearnableParameters(this)
            val = this.PrivateLayer.CustomLearnableParameters.HostValue;
            if isdlarray(val)
                val = extractdata(val);
            end
        end
        function this = set.CustomLearnableParameters(this, value)
            attributes = {...
                'size', this.PrivateLayer.CustomLearnableParametersSize,...
                'real', 'nonsparse'};
            value =  nnet.internal.cnn.layer.paramvalidation.gatherAndValidateNumericParameter(value, attributes);
            this.PrivateLayer.CustomLearnableParameters.Value = value;
        end

        function val = get.CustomLearnableParametersLearnRateFactor(this)
            val = this.getFactor(this.PrivateLayer.CustomLearnableParameters.LearnRateFactor);
        end
        function this = set.CustomLearnableParametersLearnRateFactor(this, val)
            val = gather(val);
            % iAssertValidFactor(val)
            this.PrivateLayer.CustomLearnableParameters.LearnRateFactor = this.setFactor(val);
        end

        function val = get.CustomLearnableParametersL2Factor(this)
            val = this.getFactor(this.PrivateLayer.CustomLearnableParameters.L2Factor);
        end
        function this = set.CustomLearnableParametersL2Factor(this, val)
            val = gather(val);
            % iAssertValidFactor(val)
            this.PrivateLayer.CustomLearnableParameters.L2Factor = this.setFactor(val);
        end

        function out = saveobj(this)
            out = this.saveobj@nnet.cnn.layer.LSTMLayer();
            out.CustomLearnableParameters = toStruct(this.PrivateLayer.CustomLearnableParameters);
            out.Activations = this.PrivateLayer.Activations;
            out.ActParameters = this.PrivateLayer.ActParameters;
            out.CustomLearnableParametersSize = this.PrivateLayer.CustomLearnableParametersSize;
            out.GeneralStrategy = this.PrivateLayer.GeneralStrategy;
        end
        
        function val = get.Trace(this)
            val = this.PrivateLayer.Trace;
        end
        function this = set.Trace(this, value)
            this.PrivateLayer.Trace = value;
        end

    end

    methods(Static)
        function this = loadobj(in)
            if in.Version <= 1.0
                in = iUpgradeVersionOneToVersionTwo(in);
            end
            if in.Version <= 2.0
                in = iUpgradeVersionTwoToVersionThree(in);
            end
            if in.Version <= 3.0
                in = iUpgradeVersionThreeToVersionFour(in);
            end
            if in.Version <= 4.0
                in = iUpgradeVersionFourToVersionFive(in);
            end
            internalLayer = ZLSTMLayerInternal( in.Name, ...
                in.InputSize, ...
                in.NumHiddenUnits, ...
                true, ...
                true, ...
                in.ReturnSequence, ...
                in.Activations, ...
                in.ActParameters, ...
                in.CustomLearnableParametersSize, ...
                in.GeneralStrategy, ...
                in.HasStateInputs, ...
                in.HasStateOutputs );
            internalLayer.InputWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.InputWeights);
            internalLayer.RecurrentWeights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.RecurrentWeights);
            internalLayer.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.Bias);
            internalLayer.CustomLearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.fromStruct(in.CustomLearnableParameters);
            internalLayer.CellState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter.fromStruct(in.CellState);
            internalLayer.HiddenState = nnet.internal.cnn.layer.dynamic.TrainingDynamicParameter.fromStruct(in.HiddenState);
            internalLayer.InitialHiddenState = in.InitialHiddenState;
            internalLayer.InitialCellState = in.InitialCellState;
            this = ZLSTMLayerExternal(internalLayer);
        end
    end

end