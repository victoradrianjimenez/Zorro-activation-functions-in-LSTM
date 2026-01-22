function X = zorro_tanh(X, params) %params = (slope, alpha, k)
    alpha = params(2);
    k = params(3);
    X = params(1) .* X;
    p = X < -1;
    q = X  > 1;
    if any(p, 'all')
        Xp = X(p) + 1;
        X(p) = k .* Xp ./ (1 + (k-1) .* exp(-0.5 .* Xp .* alpha)) - 1;
    end
    if any(q, 'all')
        Xq = 1 - X(q);
        X(q) = 1 - k .* Xq ./ (1 + (k-1) .* exp(-0.5 .* Xq .* alpha));
    end
end