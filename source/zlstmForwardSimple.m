function [Y, HS, CS] = zlstmForwardSimple(X, learnable, state, options, params)
% lstmForwardGeneral   Propagate Long Short-Term Memory layer forwards
%
%   LSTM forward method designed to use less memory. The 'memory' variables
%   are the full cell state and the either: the full hidden state if
%   options.ReturnLast = true, the final hidden state if options.ReturnLast
%   = false. The full cell state is required for backpropagation with
%   hard-sigmoid activation functions.
%
%       Total memory    = Y + HS + CS
%                       = (H)x(N)x(S) + (H)x(N) + (H)x(N)x(S)
%                       = (H)x(N)x(2S + 1)
%
%   [Y, HS, CS] = lstmForwardGeneral(X, learnable, state, options) computes
%   the forward propagation of the Long Short-Term Memory layer using input
%   data X, input weights W, recurrent weights R, bias term b and initial
%   cell state C0, and initial hidden units Y0.
%
%   Definitions:
%   D := Number of dimensions of the input data
%   N := Number of input observations (mini-batch size)
%   S := Sequence length
%   H := Hidden units size
%
%   Inputs:
%   X - Input data                      (D)x(N)x(S) array
%
%   learnable.W - Input weights         (4*H)x(D) matrix
%   learnable.R - Recurrent weights     (4*H)x(H) matrix
%   learnable.b - Bias                  (4*H)x(1) vector
%
%   state.c0 - Initial cell state       (H)x(1)/(H)x(N) array
%   state.y0 - Initial hidden units     (H)x(1)/(H)x(N) array
%
%   options.ReturnLast - Specify output as sequence or last
%   options.Activation - Input activation function handle
%   options.RecurrentActivation - Recurrent activation function handle
%
%   Outputs:
%   Y - Output                (H)x(N)x(S)/(H)x(N)x(1) array
%   HS - Hidden state         (H)x(N)x(1)/(H)x(N)x(S) array
%   CS - Cell state           (H)x(N)x(S) array

%   Copyright 2017 The MathWorks, Inc.

% Learnable parameters
W = learnable.W;
R = learnable.R;
b = learnable.b;

% State variables
Y0 = state.y0;
C0 = state.c0;

% Options
returnLast      = options.ReturnLast;
fRecurrentActFn = options.fRecurrentActivation;
iRecurrentActFn = options.iRecurrentActivation;
oRecurrentActFn = options.oRecurrentActivation;
zActFn          = options.zActivation;
actFn           = options.Activation;

% Determine dimensions
[~, N, S] = size(X);
H = size(R, 2);

% Pre-allocate output array and cell state
Y = zeros(H, N, S, 'like', X);
CS = zeros(H, N, S, 'like', X);

% Indexing helpers
[zInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(H);

% Split activation constants and join with learning parameters
nP = numel(params) / 5;
aP = params(1:nP);
zP = params(1+nP:2*nP);
iP = params(1+2*nP:3*nP);
fP = params(1+3*nP:4*nP);
oP = params(1+4*nP:end);

% disp(num2str(ap'));

% Forward propagate through time
for tt = 1:S
    % Linear gate operations
    G = W*X(:, :, tt) + R*Y0 + b;
    % Nonlinear gate operations
    Gz = zActFn( G(zInd, :), zP );
    Gi = iRecurrentActFn( G(iInd, :), iP );
    Gf = fRecurrentActFn( G(fInd, :), fP );
    Go = oRecurrentActFn( G(oInd, :), oP );
    % Cell state update
    CStt = Gz .* Gi + Gf .* C0;

    CS(:, :, tt) = CStt;
    % Layer output
    Y(:, :, tt) = actFn( CStt, aP) .* Go;

    % for the next iteration
    Y0 = Y(:, :, tt);
    C0 = CS(:, :, tt);
end

if returnLast
    HS = Y;
    Y = HS(:, :, end);
else
    HS = Y(:, :, end);
end
end
