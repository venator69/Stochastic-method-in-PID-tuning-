import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import warnings
import os

# Suppress some potential warnings from scipy
warnings.filterwarnings("ignore", category=UserWarning, module='scipy.signal')
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignore overflow/underflow

# Define the plant and PID simulation
def plant_response(Kp, Ki, Kd):
    """Simulates the plant with PID and returns performance metrics."""
    # Ensure positive gains
    Kp, Ki, Kd = max(0.01, Kp), max(0.01, Ki), max(0.0, Kd)

    # Plant: 1/(2s^2 + 10s + 15)
    num_plant = [1]
    den_plant = [2, 10, 15]

    # PID controller: (Kd*s^2 + Kp*s + Ki)/s
    num_pid = [Kd, Kp, Ki]
    den_pid = [1, 0]

    try:
        # Open-loop system
        num_open = np.convolve(num_pid, num_plant)
        den_open = np.convolve(den_pid, den_plant)

        # Closed-loop system: G_cl = G_pid*G_plant / (1 + G_pid*G_plant)
        max_len = max(len(num_open), len(den_open))
        num_cl = np.pad(num_open, (max_len - len(num_open), 0), 'constant')
        den_cl = np.pad(den_open, (max_len - len(den_open), 0), 'constant')
        den_cl = den_cl + num_cl # Denominator of closed-loop

        # Create LTI system object for step response
        system = signal.lti(num_cl, den_cl)
        t_sim = np.linspace(0, 15, 1500) # Simulation time
        t, y = signal.step(system, T=t_sim)

        # Check for instability / extreme values
        if np.any(np.isnan(y)) or np.max(np.abs(y)) > 100:
            return 15.0, 10.0, 1000.0 # High penalty values

        final_value = y[-1]
        if abs(final_value) < 1e-6:
            return 15.0, 1.0, 1000.0

        # Steady-state error (SSE)
        SSE = np.abs(1.0 - final_value)

        # Percent overshoot (%OS)
        max_value = np.max(y)
        OS = ((max_value - final_value) / final_value) * 100.0
        OS = max(0, OS)

        # Settling time (Ts) - 2% criterion
        settling_band_upper = final_value * 1.02
        settling_band_lower = final_value * 0.98
        Ts = t_sim[-1]
        for i in range(len(t) - 1, -1, -1):
            if not (settling_band_lower <= y[i] <= settling_band_upper):
                Ts = t[i + 1] if i + 1 < len(t) else t_sim[-1]
                break
        else:
            Ts = t[0]

        if Ts >= t_sim[-1] * 0.99: Ts = 15.0
        if SSE > 5.0 : SSE = 10.0

        return Ts, SSE, OS

    except Exception as e:
        return 15.0, 10.0, 1000.0 # High penalty values

# Generate training data
def generate_data(num_samples=10000):
    X = []
    y = []
    print(f"Generating {num_samples} data samples...")
    for i in range(num_samples):
        if (i + 1) % 1000 == 0: print(f"  Generated {i+1}/{num_samples}...")

        Kp = np.random.uniform(0.1, 30) # Wider range
        Ki = np.random.uniform(0.01, 20) # Wider range
        Kd = np.random.uniform(0.01, 15) # Wider range

        Ts, SSE, OS = plant_response(Kp, Ki, Kd)

        # Filter for "reasonable" data
        if Ts < 14.0 and SSE < 1.0 and OS < 100.0:
            X.append([Ts, SSE, OS])
            y.append([Kp, Ki, Kd])
    print(f"Generated {len(y)} valid samples.")
    return np.array(X), np.array(y)

# Define the neural network
class PIDTuner(nn.Module):
    def __init__(self):
        super(PIDTuner, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256) # Increased size
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3) # Output: Kp, Ki, Kd

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.output_relu = nn.ReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        x = self.output_relu(x) + 1e-4 # Ensure positive gains
        return x

# Train the model
def train_model(num_samples=10000, epochs=2000):
    X, y = generate_data(num_samples)

    if len(X) == 0:
        print("No valid data generated. Cannot train.")
        return None, None, None, None, None, None, None, None

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    # Normalize inputs (X)
    X_mean, X_std = X.mean(0), X.std(0)
    X_std = torch.where(X_std == 0, torch.tensor(1.0), X_std)
    X_norm = (X - X_mean) / X_std

    # Normalize outputs (y)
    y_mean, y_std = y.mean(0), y.std(0)
    y_std = torch.where(y_std == 0, torch.tensor(1.0), y_std)
    y_norm = (y - y_mean) / y_std

    model = PIDTuner()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Slightly lower LR

    print(f"\nTraining model for {epochs} epochs...")
    losses = []
    
    # --- For plotting best performance ---
    eval_interval = 50 # How often to check performance
    X_eval_target = torch.FloatTensor([2.0, 0.02, 5.0]) # A good target
    X_eval_norm = (X_eval_target - X_mean) / X_std
    
    best_Ts, best_SSE, best_OS = float('inf'), float('inf'), float('inf')
    best_Kp, best_Ki, best_Kd = 0.0, 0.0, 0.0 # Store best K values
    epochs_list, best_ts_list, best_sse_list, best_os_list = [], [], [], []
    # --- End performance tracking setup ---

    for epoch in range(epochs):
        model.train() # Set to training mode
        optimizer.zero_grad()
        outputs_norm = model(X_norm)
        loss = criterion(outputs_norm, y_norm)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # --- Evaluate and track best performance ---
        if (epoch + 1) % eval_interval == 0:
            model.eval() # Set to evaluation mode
            with torch.no_grad():
                y_pred_norm = model(X_eval_norm.unsqueeze(0))
                y_pred = y_pred_norm * y_std + y_mean
                Kp_curr, Ki_curr, Kd_curr = y_pred[0].numpy()
                
                Ts_curr, SSE_curr, OS_curr = plant_response(Kp_curr, Ki_curr, Kd_curr)

                # Update best if current is better (using a simple sum for now)
                # A more sophisticated metric could be used.
                if (Ts_curr + SSE_curr + OS_curr / 10) < (best_Ts + best_SSE + best_OS / 10):
                    best_Ts = Ts_curr
                    best_SSE = SSE_curr
                    best_OS = OS_curr
                    best_Kp, best_Ki, best_Kd = Kp_curr, Ki_curr, Kd_curr

                epochs_list.append(epoch + 1)
                best_ts_list.append(best_Ts)
                best_sse_list.append(best_SSE)
                best_os_list.append(best_OS)
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f} | '
                  f'Best Ts: {best_Ts:.2f}, SSE: {best_SSE:.3f}, OS: {best_OS:.2f}')
        elif (epoch + 1) % 200 == 0:
             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')
        # --- End evaluation ---

    # Create plot directory if it doesn't exist
    if not os.path.exists('plot'):
        os.makedirs('plot')

    # Plot Training Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot/training_loss.png') # Save the plot
    plt.close() # Close the plot to free memory

    # Plot Best Performance Over Time
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Best Achieved Performance Over Training Epochs')

    axs[0].plot(epochs_list, best_ts_list, marker='.', linestyle='-', color='b')
    axs[0].set_ylabel('Settling Time (Ts)')
    axs[0].set_title('Best Settling Time')
    axs[0].grid(True)
    axs[0].set_ylim(bottom=0, top=min(15, max(best_ts_list)*1.1 if best_ts_list else 15)) # Dynamic y-lim

    axs[1].plot(epochs_list, best_sse_list, marker='.', linestyle='-', color='g')
    axs[1].set_ylabel('Steady State Error (SSE)')
    axs[1].set_title('Best Steady State Error')
    axs[1].grid(True)
    axs[1].set_ylim(bottom=0, top=min(1, max(best_sse_list)*1.1 if best_sse_list else 1)) # Dynamic y-lim

    axs[2].plot(epochs_list, best_os_list, marker='.', linestyle='-', color='r')
    axs[2].set_ylabel('Overshoot (%OS)')
    axs[2].set_title('Best Overshoot')
    axs[2].set_xlabel('Epoch')
    axs[2].grid(True)
    axs[2].set_ylim(bottom=0, top=min(100, max(best_os_list)*1.1 if best_os_list else 100)) # Dynamic y-lim

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig('plot/best_performance.png') # Save the plot
    plt.close() # Close the plot

    return model, X_mean, X_std, y_mean, y_std, best_Kp, best_Ki, best_Kd

# Test the model
def test_model(model, X_mean, X_std, y_mean, y_std, Ts_target=2.0, SSE_target=0.05, OS_target=10.0):
    # Prepare input (with normalization)
    X_test = torch.FloatTensor([Ts_target, SSE_target, OS_target])
    X_test_norm = (X_test - X_mean) / X_std

    # Predict PID parameters
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        y_pred_norm = model(X_test_norm.unsqueeze(0)) # Add batch dimension

    # Denormalize output
    y_pred = y_pred_norm * y_std + y_mean
    Kp, Ki, Kd = y_pred[0].numpy()
    # Ensure non-negative Kd
    Kd = max(0.0, Kd)

    print(f"\n--- Testing for Ts={Ts_target}, SSE={SSE_target}, OS={OS_target} ---")
    print(f"Predicted PID: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}")

    # Verify the performance by simulating
    Ts_actual, SSE_actual, OS_actual = plant_response(Kp, Ki, Kd)
    print(f"Actual Performance: Ts={Ts_actual:.4f}s, SSE={SSE_actual:.4f}, OS={OS_actual:.4f}%")

    # Plot step response
    num_plant = [1]
    den_plant = [2, 10, 15]
    num_pid = [Kd, Kp, Ki]
    den_pid = [1, 0]
    num_open = np.convolve(num_pid, num_plant)
    den_open = np.convolve(den_pid, den_plant)
    max_len = max(len(num_open), len(den_open))
    num_cl = np.pad(num_open, (max_len - len(num_open), 0), 'constant')
    den_cl = np.pad(den_open, (max_len - len(den_open), 0), 'constant') + num_cl

    # Create plot directory if it doesn't exist
    if not os.path.exists('plot'):
        os.makedirs('plot')

    try:
        system = signal.lti(num_cl, den_cl)
        t, y = signal.step(system, T=np.linspace(0, 10, 1000))

        plt.figure()
        plt.plot(t, y, label=f'Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}')
        plt.axhline(1.0, color='r', linestyle='--', label='Setpoint')
        plt.xlabel('Time (s)')
        plt.ylabel('Response')
        plt.title(f'Step Response (Target: Ts={Ts_target}, OS={OS_target})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'plot/step_response_Ts{Ts_target}_OS{OS_target}.png') # Save the plot
        plt.close() # Close the plot

    except Exception as e:
        print(f"Could not plot response: {e}")

    return Kp, Ki, Kd

# Main execution
if __name__ == "__main__":
    trained_model, xm, xs, ym, ys, best_kp, best_ki, best_kd = train_model(num_samples=20000, epochs=1000) # Increased samples/epochs

    if trained_model:
        print(f"\n--- Best Local PID Parameters Found During Training ---")
        print(f"Best Kp = {best_kp:.4f}")
        print(f"Best Ki = {best_ki:.4f}")
        print(f"Best Kd = {best_kd:.4f}")
        
        test_model(trained_model, xm, xs, ym, ys, Ts_target=2.0, SSE_target=0.02, OS_target=5.0)
        test_model(trained_model, xm, xs, ym, ys, Ts_target=1.5, SSE_target=0.01, OS_target=10.0)
        # Add more tests if needed