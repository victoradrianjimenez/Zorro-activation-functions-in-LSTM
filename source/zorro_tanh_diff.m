function Y = zorro_tanh_diff(X, params) %params = (slope, alpha, k)
    alpha = params(2);
    k = params(3);
    slope = params(1);
    X = slope .* X;
    p = X < -1;
    q = X > 1;
    Y = zeros(size(X)) + slope;
    if any(p, 'all')
        Xp = -0.5.*X(p) - 0.5;
        G = 1 ./ (1 + (k-1) .* exp(alpha .* Xp));
        Y(p) = k .* G .* (1 - alpha .* Xp .* (1 - G)) .* slope;
    end
    if any(q, 'all')
        Xq = 0.5.*X(q) - 0.5;
        G = 1 ./ (1 + (k-1).*exp(alpha .* Xq));
        Y(q) = k .* G .* (1 - alpha .* Xq .* (1 - G)) .* slope;
    end
end