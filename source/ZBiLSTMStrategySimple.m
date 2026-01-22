classdef ZBiLSTMStrategySimple < nnet.internal.cnn.layer.util.ExecutionStrategy
    % BiLSTMSimpleStrategy   Execution strategy for running BiLSTM with a
    % general set of options. This strategy is used for both host and GPU
    % implementations.
    
    %   Copyright 2018-2020 The MathWorks, Inc.
    
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
        function this = ZBiLSTMStrategySimple(activations, params, returnSequence)
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
            % Split data into forward/backward sequences
            [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b);
            [c0f, y0f, c0b, y0b] = iSplitStates(c0, y0);
            
            % Forward sequence
            [learnablef, statef] = iLearnableAndStateStruct(Wf, Rf, bf, y0f, c0f);
            learnablef.ap = ap(1:end/2, :);
            [Yf, Hf, Cf] = zlstmForwardSimple(X, learnablef, statef, this.OptionsF, this.CustomParams);
            
            % Backward sequence
            [learnableb, stateb] = iLearnableAndStateStruct(Wb, Rb, bb, y0b, c0b);
            learnableb.ap = ap(1+end/2:end, :);
            [Yb, Hb, Cb] = zlstmForwardSimple(flip(X, 3), learnableb, stateb, this.OptionsF, this.CustomParams);
            
            % Concatenate outputs
            Y = cat(1, Yf, flip(Yb, 3));
            H = cat(1, Hf, Hb);
            C = cat(1, Cf, Cb);
            
            % Allocate memory
            memory.HiddenState = H;
            memory.CellState = C;
        end
        
        function [dX, dW, dR, dB, dAP, dy0, dc0] = backward(this, X, W, R, b, ap, c0, y0, Y, memory, dY)
            % Split data into forward/backward sequences
            [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b);
            [c0f, y0f, c0b, y0b] = iSplitStates(c0, y0);
            [Cf, Hf, Cb, Hb] = iSplitStates(memory.CellState, memory.HiddenState);
            [Yf, Yb] = iSplitAcrossFirstDimension(Y);
            [dYf, dHf, dCf, dYb, dHb, dCb] = iSplitGradients(dY);
            
            % Forward sequence
            [learnablef, statef] = iLearnableAndStateStruct(Wf, Rf, bf, y0f, c0f);
            learnablef.ap = ap(1:end/2, :);
            [outputsf, derivativesf] = iOutputsAndDerivativesStruct(Yf, Hf, Cf, dYf, dHf, dCf);
            [dXf, dWeightsf{1:(nargout-1)}] =  zlstmBackwardSimple(X, learnablef, statef, outputsf, derivativesf, this.OptionsB, this.CustomParams);

            % Backward sequence
            [learnableb, stateb] = iLearnableAndStateStruct(Wb, Rb, bb, y0b, c0b);
            learnableb.ap = ap(1+end/2:end, :);
            [outputsb, derivativesb] = iOutputsAndDerivativesStruct(flip(Yb, 3), Hb, Cb, flip(dYb, 3), dHb, dCb);
            [dXb, dWeightsb{1:(nargout-1)}] = zlstmBackwardSimple(flip(X, 3), learnableb, stateb, outputsb, derivativesb, this.OptionsB, this.CustomParams);
            
            % Concatenate outputs
            dX = dXf + flip(dXb, 3);
            needsWeightGradients = nargout > 1;
            if needsWeightGradients
                % Concatenate weights derivatives
                [dWf, dRf, dBf, dAPf] = deal( dWeightsf{1:4} );
                [dWb, dRb, dBb, dAPb] = deal( dWeightsb{1:4} );
                dW = cat(1, dWf, dWb);
                dR = cat(1, dRf, dRb);
                dB = cat(1, dBf, dBb);
                dAP = cat(1, dAPf, dAPb);
                
                needsStateGradients = nargout > 5;
                if needsStateGradients
                    % Concatenate state derivatives
                    [dh0f, dc0f] = deal( dWeightsf{5:6} );
                    [dh0b, dc0b] = deal( dWeightsb{5:6} );
                    dy0 = cat(1, dh0f, dh0b);
                    dc0 = cat(1, dc0f, dc0b);
                end
            end
        end
    end
end

%% Data manipulation
function [Wf, Rf, bf, Wb, Rb, bb] = iSplitWeights(W, R, b)
[Wf, Wb] = iSplitAcrossFirstDimension( W );
[Rf, Rb] = iSplitAcrossFirstDimension( R );
[bf, bb] = iSplitAcrossFirstDimension( b );
end

function [c0f, y0f, c0b, y0b] = iSplitStates(c0, y0)
[c0f, c0b] = iSplitAcrossFirstDimension( c0 );
[y0f, y0b] = iSplitAcrossFirstDimension( y0 );
end

function [dYf, dHf, dCf, dYb, dHb, dCb] = iSplitGradients(dY)
if iscell(dY)
    [dYf, dYb] = iSplitAcrossFirstDimension( dY{1} );
    [dHf, dHb] = iSplitAcrossFirstDimension( dY{2} );
    [dCf, dCb] = iSplitAcrossFirstDimension( dY{3} );
else
    [dYf, dYb] = iSplitAcrossFirstDimension( dY );
    [h, n] = size(dY, 1, 2);
    h = 0.5.*h;
    dHf = zeros( [h, n], 'like', dY );
    dHb = zeros( [h, n], 'like', dY );
    dCf = zeros( [h, n], 'like', dY );
    dCb = zeros( [h, n], 'like', dY );
end
end

function [Zf, Zb] = iSplitAcrossFirstDimension( Z )
H = 0.5*size(Z, 1);
fInd = 1:H;
bInd = H + fInd;
Zf = Z(fInd, :, :);
Zb = Z(bInd, :, :);
end

function [learnable, state] = iLearnableAndStateStruct(W, R, b, y0, c0)
learnable.W = W;
learnable.R = R;
learnable.b = b;
state.y0 = y0;
state.c0 = c0;
end

function [outputs, derivatives] = iOutputsAndDerivativesStruct(Y, H, C, dY, dH, dC)
outputs.Y = Y;
outputs.HS = H;
outputs.CS = C;
derivatives.dZ = dY;
derivatives.dHS = dH;
derivatives.dCS = dC;
end
