function J_allow_from_linear_rule_ms
%% Max allowable jitter from L + J < beta with alpha = 1
% Workflow:
% 1) Compute beta = J_max at L = 0 by simulation (uniform delay in [0, J]).
% 2) For fixed h = 10 ms and L in 2:2:20 ms, compute J_allow(L) = max(0, beta - L).
%
% All inputs/outputs shown in milliseconds; Jitterbug internals in seconds.
%
% Requires on path: initjitterbug, addtimingnode, addcontsys, adddiscsys,
%                   calcdynamics, calccost, lqgdesign.

clear; clc; close all;

%% 0) Settings (ms)
dt_ms      = 0.5;          % Jitterbug tick [ms]
h_ms       = 6.0;         % FIXED sampling period [ms]
L_grid_ms  = 0:2:38;       % L = 2,4,...,20 ms
j_guess    = 0.6;          % initial hi ~ j_guess*h
bins       = 80;           % discretization bins for delay distribution
tol_ms     = 0.25;         % binary-search tolerance on jitter [ms]
max_iter   = 22;           % binary-search iterations
use_tau_comp = false;      % compensation tau = J/2 (false = conservative)

%% Convert to seconds for Jitterbug
dt_s = dt_ms/1000;  h_s = h_ms/1000;

%% 1) Plant & cost (edit to your plant if needed)
s = tf('s');

J = 0.01; b = 0.1; K = 0.01; R = 1; L = 0.5;
P = K / ( (J*s + b)*(L*s + R) + K^2 );

%P = 1000/(s^2 + s);    % example plant (like examples/distributed.m)
Q  = diag([1 1]);      % cost on [y; u]
R1 = 1;                % process noise intensity (cont-time)
R2 = 0.01;             % measurement noise variance (discrete-time)

%% 2) Compute beta = J_max at L = 0  (alpha = 1 case)
beta_ms = find_Jmax_at_L0_ms(P,Q,R1,R2,dt_s,h_s,j_guess,bins,tol_ms,max_iter,use_tau_comp);
fprintf('Estimated beta (J_max at L=0): %.4f ms (alpha = 1)\n', beta_ms);

%% 3) Compute allowable jitter from the linear rule: J_allow(L) = max(0, beta - L)
J_allow_ms = max(0, beta_ms - L_grid_ms);

%% 4) Print arrays and a 2-col table
disp('--- L (ms) ---');        disp(L_grid_ms);
disp('--- J_allow (ms) ---');  disp(J_allow_ms);
disp('--- [L_ms , J_allow_ms] ---');  disp([L_grid_ms(:), J_allow_ms(:)]);

%% 5) Plot J_allow vs L and the line L + J = beta
figure; hold on; grid on;
%plot(L_grid_ms, J_allow_ms, 'o-','LineWidth',1.6, 'DisplayName','J_{allow}(L) = \beta - L');

plot(L_grid_ms, J_allow_ms, 'o-','LineWidth',1.6);

% Plot the boundary line in the (L,J) plane
L_plot = linspace(min(L_grid_ms), max(L_grid_ms), 200);
J_line = max(0, beta_ms - L_plot);
%%plot(L_plot, J_line, '--','LineWidth',1.3, 'DisplayName','L + J = \beta');
xlabel('Nominal delay L [ms]'); ylabel('Jitter J [ms]');
title(sprintf('Allowable jitter from L + J < \\beta (h = %.2f ms, \\alpha = 1)', h_ms));
%legend('Location','northeast');

end % ===== end main =====

%% ===== Helper: find J_max at L = 0 (uniform delay in [0, J]) =====
function beta_ms = find_Jmax_at_L0_ms(P,Q,R1,R2,dt_s,h_s,j_guess,bins,tol_ms,max_iter,use_tau_comp)
    % Expand upper bound and binary-search for J_max at L=0.
    h_ms = h_s*1000;
    lo = 0;  hi = j_guess*h_ms;
    [ok_hi,~] = test_pair_L0_ms(P,Q,R1,R2,dt_s,h_s,hi,bins,use_tau_comp);
    grow = 0;
    while ok_hi && hi < 3*h_ms && grow < 6
        hi = hi*1.8 + 1e-6;
        [ok_hi,~] = test_pair_L0_ms(P,Q,R1,R2,dt_s,h_s,hi,bins,use_tau_comp);
        grow = grow + 1;
    end
    bestJ = NaN;
    for it = 1:max_iter
        mid = 0.5*(lo + hi);
        [ok_mid,Jmid] = test_pair_L0_ms(P,Q,R1,R2,dt_s,h_s,mid,bins,use_tau_comp);
        if ok_mid, lo = mid; bestJ = Jmid; else, hi = mid; end
        if (hi - lo) <= tol_ms, break; end
    end
    beta_ms = lo;    % by definition: beta = J_max at L=0 (alpha = 1)
end

%% ===== Helper: simulate (L = 0, jitter = [0, J]) in ms =====
function [is_ok, Jcost] = test_pair_L0_ms(P,Q,R1,R2,dt_s,h_s,J_ms,bins,use_tau_comp)
    % Build Ptau for L=0: uniform on [0, J]
    Ptau = mkPtau_ms(J_ms, dt_s, bins);

    % LQG controller; optional compensation tau = J/2 (seconds)
    tau = double(use_tau_comp) * (J_ms/1000)/2;
    C = lqgdesign(P, Q, R1, R2, h_s, tau);

    % Jitterbug network
    N = initjitterbug(dt_s, h_s);
    N = addtimingnode(N, 1, Ptau, 2);   % periodic node with delay ~ U[0, J]
    N = addtimingnode(N, 2);            % controller node

    Samp = ss([], [], [], 1, h_s);      % DT sampler (gain=1, Ts=h)
    Rsam = 0;

    N = addcontsys(N, 1, P, 3);
    N = adddiscsys(N, 2, Samp, 1, 1, [], Rsam);
    N = adddiscsys(N, 3, C,    2, 2);
    N = calcdynamics(N);

    Jcost = calccost(N);
    is_ok = isfinite(Jcost) && Jcost < 1e10;
end

%% ===== Build Ptau for L = 0 (inputs in ms; grid in seconds) =====
function Ptau = mkPtau_ms(J_ms, dt_s, bins)
    J_s = max(0, J_ms/1000);
    if J_s <= 0
        Ptau = 1;   % deterministic zero delay
        return;
    end
    nJ = max(1, min(bins, round(J_s/dt_s) + 1));  % bins covering [0, J]
    Ptau = ones(1, nJ);
    Ptau = Ptau / sum(Ptau);
end
