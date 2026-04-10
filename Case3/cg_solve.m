function [x, rel_res_history, num_iter] = cg_solve(A, b, tol, max_iter)
% CG_SOLVE  Conjugate Gradient solver for symmetric positive definite systems Ax = b.
%           Based on Algorithm 6.18 from Saad's "Iterative Methods for Sparse
%           Linear Systems", 2nd edition.
%
% Inputs:
%   A        - (N x N) symmetric positive definite matrix
%   b        - (N x 1) right-hand side vector
%   tol      - convergence tolerance on relative residual norm
%   max_iter - maximum number of iterations allowed
%
% Outputs:
%   x               - (N x 1) approximate solution
%   rel_res_history - relative residual norm ||r_k|| / ||r_0|| at each iteration
%   num_iter        - number of iterations performed

    N = length(b);
    x = zeros(N, 1);          % initial guess x_0 = 0

    r = b - A * x;            % r_0 = b - A*x_0  (= b since x_0 = 0)
    p = r;                    % p_0 = r_0

    rtr     = r' * r;         % scalar: r_0^T r_0
    r0_norm = sqrt(rtr);      % ||r_0||, used for relative residual

    rel_res_history = zeros(max_iter, 1);

    for k = 1 : max_iter

        Ap    = A * p;
        pAp   = p' * Ap;

        alpha = rtr / pAp;            % step length

        x = x + alpha * p;            % update solution
        r = r - alpha * Ap;           % update residual

        rtr_new = r' * r;

        rel_res = sqrt(rtr_new) / r0_norm;
        rel_res_history(k) = rel_res;

        if rel_res < tol
            num_iter = k;
            rel_res_history = rel_res_history(1:k);
            return
        end

        beta = rtr_new / rtr;         % Fletcher-Reeves direction update
        p    = r + beta * p;          % update search direction

        rtr  = rtr_new;
    end

    % Reached max_iter without converging
    num_iter = max_iter;
    fprintf('[CG] Warning: did not converge in %d iterations. Final rel. res = %.2e\n', ...
            max_iter, rel_res_history(end));
end
