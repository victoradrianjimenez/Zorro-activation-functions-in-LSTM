function Y = zorro_sigm_diff(X, params) %params = (slope, alpha, k)
    alpha = params(2);
    k = params(3);
    slope = params(1);
    X = slope .* X + 0.5;
    p = X < 0;
    q = X > 1;
    Y = zeros(size(X)) + slope;
    if any(p, 'all')
        Xp = X(p);
        G = 1 ./ (1 + (k-1) .* exp(-alpha .* Xp));
        Y(p) = k .* G .* (1 + alpha .* Xp .* (1 - G)) .* slope;
    end
    if any(q, 'all')
        Xq = 1 - X(q);
        G = 1 ./ (1 + (k-1).*exp(-alpha .* Xq));
        Y(q) = k .* G .* (1 + alpha .* Xq .* (1 - G)) .* slope;
    end
end