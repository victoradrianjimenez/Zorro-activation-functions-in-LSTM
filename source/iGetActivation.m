function act = iGetActivation( activation )
act.dpFcn = @(x, ~) zeros(size(x));
switch activation
    case 'tanh'
        act.Fcn = @iTanh;
        act.dFcn = @iTanhDiff;
    case 'softsign'
        act.Fcn = @iSoftSign;
        act.dFcn = @iSoftSignDiff;
    case 'sigmoid'
        act.Fcn = @iSigmoid;
        act.dFcn = @iSigmoidDiff;
    case 'hard-sigmoid'
        act.Fcn = @hardSigmoidForward;
        act.dFcn = @hardSigmoidBackward;
    otherwise
        act.Fcn = eval(strcat('@', activation));
        act.dFcn = eval(strcat('@', activation, '_diff'));
        try
            act.dpFcn = eval(strcat('@', activation, '_dparam'));
        catch            
        end
end
end

%% Activation functions and derivatives
function y = iSigmoid(x, ~)
y = 1./(1 + exp(-x));
end

function y = iSigmoidDiff(x, ~)
y = ( 0.5.*sech(0.5.*x) ).^2;
end

function y = iTanh(x, ~)
y = tanh(x);
end

function y = iTanhDiff(x, ~)
y = sech(x).^2;
end

function y = iSoftSign(x, ~)
y = x./(1 + abs(x));
end

function y = iSoftSignDiff(x, ~)
y = (1 + abs(x)).^-2;
end

function Y = hardSigmoidForward(X, ~)
Y = max( 0, min(1, 0.2.*X + 0.5, 'includenan'), 'includenan' );
end

function dY = hardSigmoidBackward( X, ~)
dY = zeros( size(X), 'like', X );
dY( (X >= -2.5) & (X <= 2.5) ) = 0.2;
end
