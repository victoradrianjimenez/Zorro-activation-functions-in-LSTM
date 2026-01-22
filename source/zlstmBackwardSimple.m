function [dX, dW, dR, db, ds, dHS, dCS] = zlstmBackwardSimple(X, learnable, state, outputs, derivatives, options, params)
% lstmBackwardGeneral   Propagate Long Short-Term Memory layer backwards
%
%   LSTM backward method designed to use less memory. The 'memory' variables
%   are the full cell state and the either: the full hidden state if
%   options.ReturnLast = true, the final hidden state if options.ReturnLast
%   = false. The full cell state is required for backpropagation with
%   hard-sigmoid activation functions.
%
%       Total memory    = Y + HS + CS
%                       = (H)x(N)x(S) + (H)x(N) + (H)x(N)x(S)
%                       = (H)x(N)x(2S + 1)
%
%   [dX, dW, dR, db] = lstmBackwardGeneral(X, learnable, state, outputs, dZ, options)
%   computes the backward propagation of the Long Short-Term Memory layer
%   using input data X, input weights W, recurrent weights R, bias term b,
%   initial cell state c0, initial hidden units y0, output data Y, cell
%   state C, layer gates G and output derivative dZ.
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
%   outputs.Y - Forward output          (H)x(N)x(S)/(H)x(N)x(1) array
%   outputs.HS - Hidden state           (H)x(N)x(1)/(H)x(N)x(S) array
%   outputs.CS - Cell state             (H)x(N)x(S) array
%
%   derivatives.dZ - Next layer derivative          (H)x(N)x(S)/(H)x(N)x(1) array
%   derivatives.dHS - Hidden state derivative       (H)x(N)x(1) array
%   derivatives.dCS - Cell state derivative         (H)x(N)x(1) array
%
%   options.ReturnLast - Specify output as sequence or last
%   options.Activation - Input activation function handle
%   options.dActivation - Input activation derivative function handle
%   options.RecurrentActivation - Recurrent activation function handle
%   options.dRecurrentActivation - Recurrent activation derivative function handle
%   
%   Outputs:
%   dX - Input data derivative            (D)x(N)x(S) array
%   dW - Input weights derivative         (4*H)x(D) array
%   dR - Recurrent weights derivative     (4*H)x(H) array
%   db - Bias derivative                  (4*H)x(1) vector
%   dCS - Initial cell state derivative   (H)x(N) array
%   dHS - Initial hidden state derivative (H)x(N) array
%
%   Note that dCS and dHS will always be of dimension (H)x(N), whereas the
%   initial states c0 and y0 can be specified as (H)x(1). When this is the
%   case, it is to be understood that a single state c0, say, (H)x(1) is
%   broadcast to all observations (N) that are passed into the method for a
%   forward pass. In other words, we can interpret the "true" c0Batch as
%   being c0Batch = repmat(c0, [1 N]).

%   Copyright 2017-2019 The MathWorks, Inc.

% Learnable parameters
W = learnable.W;
R = learnable.R;
b = learnable.b;
ap = learnable.ap;

% State variables
Y0 = state.y0;
C0 = state.c0;

% Forward pass outputs
Y = outputs.Y;
HS = outputs.HS;
CS = outputs.CS;

% Derivatives
dZ = derivatives.dZ;
dHS = derivatives.dHS;
dCS = derivatives.dCS;

% Options
returnLast = options.ReturnLast;

% AzIndctivation function handles
zact     = options.zActivation;
isig     = options.iRecurrentActivation;
fsig     = options.fRecurrentActivation;
osig     = options.oRecurrentActivation;
act      = options.Activation;

dzact    = options.dzActivation;
disig    = options.diRecurrentActivation;
dfsig    = options.dfRecurrentActivation;
dosig    = options.doRecurrentActivation;
dact     = options.dActivation;

% Input dimensionality
[~, N, S] = size(X);

% Indexing helpers
H = size(R, 2);
[zInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(H);

% Initialize derivatives
dG = zeros(4*H, N, 'like', X);
dX = zeros(size(X), 'like', X);
dW = zeros(size(W), 'like', X);
dR = zeros(size(R), 'like', X);
db = zeros(size(b), 'like', b);
ds = zeros(size(ap), 'like', X);

% Split activation constants and join with learning parameters
nP = numel(params) / 5;
aP = params(1:nP);
zP = params(1+nP:2*nP);
iP = params(1+2*nP:3*nP);
fP = params(1+3*nP:4*nP);
oP = params(1+4*nP:end);

% If y0 is passed as a vector, expand over batch dimension
if size(Y0, 2) == 1
    Y0 = repmat(Y0, 1, N);
end

if returnLast
    % Expand dZ over all time steps
    dZ(:, :, S) = dZ;
    dZ(:, :, 1:S-1) = 0;
    % Make sure Y is defined over all time steps
    Y = HS;
end
Rt = R';
Wt = W';
Ctt = CS(:, :, S);

%%% Backpropagation through time
for tt = S:-1:2
    Cpp = CS(:, :, tt - 1);
    Ypp = Y(:, :, tt - 1);

    % Linear gates
    G = W*X(:, :, tt) + R*Ypp + b;
    Go = G(oInd, :);
    Gf = G(fInd, :);
    Gz = G(zInd, :);
    Gi = G(iInd, :);
    
    % Layer output derivative
    dY = dZ(:, :, tt) + dHS;

    % Cell derivative
    dC = dY .* osig(Go, oP) .* dact(Ctt, aP) + dCS;

    % Output gate derivativedGi
    dG(oInd, :) = dY.*act(Ctt, aP).*dosig(Go, oP);

    % Forget gate derivative
    dG(fInd, :) = dC.*Cpp.*dfsig(Gf, fP);

    % Input gate derivative
    dG(iInd, :) = dC.*zact(Gz, zP).*disig(Gi, iP);

    % Layer input derivative
    dG(zInd, :) = dC.*isig(Gi, iP) .* dzact(Gz, zP);

    % Input data derivative
    dX(:, :, tt) = Wt * dG;

    % Input weights derivative
    dW = dW + dG*X(:, :, tt)';
    
    % Recurrent weights derivative
    dR = dR + dG*Ypp';
    
    % Bias derivative
    db = db + sum(dG, 2);

    % For the next iteration
    dHS = Rt * dG;
    dCS = dC .* fsig(Gf, fP);
    Ctt = Cpp;
end

% Linear gates
G = W*X(:, :, 1) + R*Y0 + b;
Go = G(oInd, :);
Gf = G(fInd, :);
Gz = G(zInd, :);
Gi = G(iInd, :);

% Layer output derivative
dY = dZ(:, :, 1) + dHS;

% Cell derivative
dC = dY .* osig(Go, oP) .* dact(Ctt, aP) + dCS;

% Output gate derivativedGi
dG(oInd, :) = dY.*act(Ctt, aP).*dosig(Go, oP);

% Forget gate derivative
dG(fInd, :) = dC.*C0.*dfsig(Gf, fP);

% Input gate derivative
dG(iInd, :) = dC.*zact(Gz, zP).*disig(Gi, iP);

% Layer input derivative
dG(zInd, :) = dC.*isig(Gi, iP) .* dzact(Gz, zP);

% Input data derivative
dX(:, :, 1) = Wt * dG;

% Input weights derivative
dW = dW + dG*X(:, :, 1)';

% Recurrent weights derivative
dR = dR + dG*Y0';

% Bias derivative
db = db + sum(dG, 2);

% Initial hidden state derivative and nitial cell state derivative
dHS = Rt * dG;
dCS = dC .* fsig(Gf, fP);

end
