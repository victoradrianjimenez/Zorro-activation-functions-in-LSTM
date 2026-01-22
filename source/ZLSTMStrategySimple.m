classdef ZLSTMStrategySimple < nnet.internal.cnn.layer.util.ExecutionStrategy
    % LSTMSimpleStrategy   Execution strategy for running LSTM with a
    % general set of options. This strategy is used for both host and GPU
    % implementations.
    
    %   Copyright 2018-2019 The MathWorks, Inc.
    
    properties
        % Options   Struct containing five fields:
        %           - Activation
        %           - dActivation
        %           - RecurrentActivation
        %           - dRecurrentActivation
        %           - ReturnLast
        OptionsF
        OptionsB
        CustomParams
    end
    
    methods
        function this = ZLSTMStrategySimple(activations, params, returnSequence)
            % Determine activation structs
            activationStruct = iGetActivation( activations(1) );
            zActivationStruct = iGetActivation( activations(2) );
            iRecurrentActivationStruct = iGetActivation( activations(3) );
            fRecurrentActivationStruct = iGetActivation( activations(4) );
            oRecurrentActivationStruct = iGetActivation( activations(5) );
            
            % Gather options struct
            optionsF.zActivation = zActivationStruct.Fcn;
            optionsF.iRecurrentActivation = iRecurrentActivationStruct.Fcn;
            optionsF.fRecurrentActivation = fRecurrentActivationStruct.Fcn;
            optionsF.oRecurrentActivation = oRecurrentActivationStruct.Fcn;
            optionsF.Activation = activationStruct.Fcn;
            
            optionsB.zActivation = zActivationStruct.Fcn;
            optionsB.iRecurrentActivation = iRecurrentActivationStruct.Fcn;
            optionsB.fRecurrentActivation = fRecurrentActivationStruct.Fcn;
            optionsB.oRecurrentActivation = oRecurrentActivationStruct.Fcn;
            optionsB.Activation = activationStruct.Fcn;

            optionsB.dzActivation = zActivationStruct.dFcn;
            optionsB.diRecurrentActivation = iRecurrentActivationStruct.dFcn;
            optionsB.dfRecurrentActivation = fRecurrentActivationStruct.dFcn;
            optionsB.doRecurrentActivation = oRecurrentActivationStruct.dFcn;
            optionsB.dActivation = activationStruct.dFcn;

            optionsF.ReturnLast = ~returnSequence;
            optionsB.ReturnLast = ~returnSequence;

            this.OptionsF = optionsF;
            this.OptionsB = optionsB;
            this.CustomParams = params;
        end
        
        function [Y, memory] = forward(this, X, W, R, b, ap, c0, y0)
            learnable.W = W;
            learnable.R = R;
            learnable.b = b;
            learnable.ap = ap;
            state.c0 = c0;
            state.y0 = y0;
            [Y, HS, CS] = zlstmForwardSimple(X, learnable, state, this.OptionsF, this.CustomParams);
            memory.HiddenState = HS;
            memory.CellState = CS;
        end
        
        function varargout = backward(this, X, W, R, b, ap, c0, y0, Y, memory, dZ)
            learnable.W = W;
            learnable.R = R;
            learnable.b = b;
            learnable.ap = ap;
            state.c0 = c0;
            state.y0 = y0;
            outputs.Y = Y;
            outputs.HS = memory.HiddenState;
            outputs.CS = memory.CellState;
            if iscell(dZ)
                derivatives.dZ = dZ{1};
                derivatives.dHS = dZ{2};
                derivatives.dCS = dZ{3};
            else
                derivatives.dZ = dZ;
                derivatives.dHS = iInitializeStateDerivatives( memory.HiddenState );
                derivatives.dCS = iInitializeStateDerivatives( memory.CellState );
            end
            [ varargout{1:nargout} ] =  zlstmBackwardSimple(X, learnable, state, outputs, derivatives, this.OptionsB, this.CustomParams);
        end
    end
end

function ds = iInitializeStateDerivatives( s )
ds = zeros( size(s(:, :, end)), 'like', s );
end
