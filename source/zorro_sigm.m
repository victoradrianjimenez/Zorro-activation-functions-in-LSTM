function X = zorro_sigm(X, params) %params = (slope, alpha, k)
    alpha = params(2);
    k = params(3);
    X = params(1) .* X + 0.5;
    p = X < 0;
    q = X > 1;
    if any(p, 'all')
        Xp = X(p);
        X(p) = k .* Xp ./ (1 + (k-1) .* exp(-Xp .* alpha));
    end
    if any(q, 'all')
        Xq = 1 - X(q);
        X(q) = 1 - k .* Xq ./ (1 + (k-1) .* exp(-Xq .* alpha));
    end
end