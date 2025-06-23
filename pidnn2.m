%% PIDNN Controller with Uncompensated System Comparison
% Control System: 2nd Order Plant
% Controller: Adaptive PID Neural Network

clc; clear; close all;

%% 1. System Definition and Initialization
% Continuous-time plant
plant = tf(1, [0.25 1 1]);

% Discretization
Ts = 0.1;                   % Sampling time (s)
P_z = c2d(plant, Ts, 'zoh'); % Discrete plant
[num, den] = tfdata(P_z, 'v');

% Simulation parameters
T_sim = 50;                 % Simulation duration (s)
N = round(T_sim/Ts);        % Number of samples
time = (0:N-1)*Ts;          % Time vector

% Reference signal (step at t=5s)
ref = zeros(size(time));
ref(time >= 5) = 1;

%% 2. PIDNN Controller Initialization
% Initial controller gains
Kp = 2;       % Proportional
Ki = 0.8;       % Integral
Kd = 0.15;      % Derivative

% Neural network weights
Kj1 = 1.2;      % Proportional error weight
Kj2 = 1.1;      % Integral error weight

% Learning parameters
learning_rate = 0.002;      % Main learning rate
nn_learning_rate = 0.0001;  % Neural network learning rate
momentum = 0.1;             % Momentum factor

% Control limits
u_limits = [-10 10];        % Control signal saturation
int_limits = [-5 5];        % Integral anti-windup limits

%% 3. Simulation Setup
% Initialize arrays
y_pidnn = zeros(size(time)); % PIDNN output
y_uncomp = zeros(size(time)); % Uncompensated output
u_out = zeros(size(time));   % Control signal

% Initialize states
e_prev = 0;                 % Previous error
x_int = 0;                  % Integral state

% Plant memory buffers
u_mem = zeros(1, length(num));
y_mem = zeros(1, length(den)-1);

%% 4. Main Simulation Loop
fprintf('Running simulation...\n');
for k = 1:length(time)
    % Current reference and error
    r = ref(k);
    e = r - y_pidnn(max(1,k-1)); % Use previous output
    
    % PIDNN Calculations
    v = Kj1 * e;                    % Proportional component
    x = x_int + Kj2 * e * Ts;       % Integral component
    x = max(min(x, int_limits(2)), int_limits(1)); % Anti-windup (FIXED LINE)
    w = (e - e_prev)/Ts;            % Derivative component
    
    % Control signal calculation
    u = Kp*v + Ki*x + Kd*w;
    u = max(min(u, u_limits(2)), u_limits(1));
    
    % Update controller states
    e_prev = e;
    x_int = x;
    
    % Plant simulation (PIDNN)
    u_mem = [u, u_mem(1:end-1)];
    y_temp = sum(num .* u_mem) - sum(den(2:end) .* y_mem);
    y_pidnn(k) = y_temp / den(1);
    y_mem = [y_pidnn(k), y_mem(1:end-1)];
    
    % Store control signal
    u_out(k) = u;
    
% Simulate uncompensated system
if k == 1
    y_uncomp(k) = 0;
else
    y_temp_uncomp = lsim(P_z, ref(1:k), time(1:k));
    y_uncomp(k) = y_temp_uncomp(end);
end

    
    % Adaptive learning (every 10 steps)
    if mod(k,10) == 0
        % Simplified learning rule
        delta_Kp = learning_rate * e * v;
        delta_Ki = 2 * learning_rate * e * x; % Emphasize integral
        delta_Kd = learning_rate * e * w;
        
        % Update gains with momentum
        Kp = Kp + delta_Kp + momentum*delta_Kp;
        Ki = Ki + delta_Ki + momentum*delta_Ki;
        Kd = Kd + delta_Kd + momentum*delta_Kd;
        
        % Apply constraints
        Kp = max(min(Kp, 5), 0.1);
        Ki = max(min(Ki, 2), 0.01);
        Kd = max(min(Kd, 1), 0);
    end
end

%% 5. Performance Analysis
% Calculate metrics for both systems
pidnn_info = stepinfo(y_pidnn, time, 1);
uncomp_info = stepinfo(y_uncomp, time, 1);

metrics = {
    "SSE", sum((ref-y_pidnn).^2), sum((ref-y_uncomp).^2);
    "Settling Time (2%)", pidnn_info.SettlingTime, uncomp_info.SettlingTime;
    "Overshoot (%)", pidnn_info.Overshoot, uncomp_info.Overshoot;
    "Rise Time (s)", pidnn_info.RiseTime, uncomp_info.RiseTime;
    };

%% 6. Visualization
figure('Position', [100 100 1000 700]);

% Main response plot
subplot(2,1,1);
plot(time, ref, 'k--', 'LineWidth', 1.5); hold on;
plot(time, y_pidnn, 'b', 'LineWidth', 2);
plot(time, y_uncomp, 'r', 'LineWidth', 1.5);
title('System Step Response Comparison');
xlabel('Time (s)');
ylabel('Output');
legend('Reference', 'PIDNN Controlled', 'Uncompensated', 'Location', 'southeast');
grid on;
ylim([-0.1 1.2]);

% Control signal plot
subplot(2,1,2);
plot(time, u_out, 'm', 'LineWidth', 1.5);
title('Control Signal');
xlabel('Time (s)');
ylabel('Control Effort');
grid on;

%% 7. Display Performance Metrics
fprintf('\n=== Performance Metrics ===\n');
fprintf('%-20s %-15s %-15s\n', 'Metric', 'PIDNN', 'Uncompensated');
fprintf('--------------------------------------------\n');
for i = 1:size(metrics,1)
    fprintf('%-20s %-15.4f %-15.4f\n', metrics{i,1}, metrics{i,2}, metrics{i,3});
end