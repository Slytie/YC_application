import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# DeepNet class definition
class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DeepNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, input_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Bisection method definition
def bisection(f, a, b, tol=1e-6, max_iter=100):
    fa = f(a)
    fb = f(b)
    assert (fa * fb < 0).flatten().all(), "The function must have different signs at a and b."
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        mask = fa * fc < 0
        a = torch.where(mask, a, c)
        b = torch.where(mask, c, b)
        fa = torch.where(mask, fa, fc)
        fb = torch.where(mask, fc, fb)
        if torch.max(torch.abs(b - a)) < tol:
            break
    return (a + b) / 2

# DeepCrankNicolsonNetC class definition
class DeepCrankNicolsonNetC:
    def __init__(self, input_size, hidden_size, num_layers, delta_t, n_steps, newton_iterations=10, lr=0.1, reg_factor=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DeepNet(input_size, hidden_size, num_layers).to(self.device)
        self.delta_t = delta_t
        self.n_steps = n_steps
        self.newton_iterations = newton_iterations
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.reg_factor = reg_factor

    def forward(self, X0):
        X = X0
        X_outputs = [X0]
        for _ in range(self.n_steps):
            X = X + self.delta_t * self.net(X)
            X_outputs.append(X)
        return torch.stack(X_outputs)

    def backward(self, Xn):
        X = Xn
        X_outputs = [Xn]
        for _ in range(self.n_steps):
            X_prev = X
            for _ in range(self.newton_iterations):
                f_X = self.net(X)
                X = bisection(
                    f=lambda x: x - X_prev - self.delta_t * 0.5 * (self.net(X_prev) + self.net(x)),
                    a=X_prev - 10, b=X_prev + 10, tol=1e-6, max_iter=100
                )
            X_outputs.append(X)
        return torch.stack(X_outputs[::-1])

    def compute_loss(self, X0_seq, Xn_seq):
        X_outputs_forward = self.forward(X0_seq)
        Xn_seq.requires_grad_(True)
        X_outputs_backward = self.backward(Xn_seq)

        # Compute the loss at each time step for forward and backward approximations
        loss_forward = sum(self.criterion(X_outputs_forward[i], X0_seq) for i in range(self.n_steps + 1))
        loss_backward = sum(self.criterion(X_outputs_backward[i], Xn_seq) for i in range(self.n_steps + 1))

        reg_term = self.reg_factor * sum(p.norm() for p in self.net.parameters())
        return loss_forward + loss_backward + reg_term

    def train(self, X0_seq, Xn_seq, X0_val_seq, Xn_val_seq, epochs=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.compute_loss(X0_seq, Xn_seq)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

# Training function
def train_model(X0_train, Xn_train, X0_val, Xn_val, epochs=100):
    # Hyperparameters
    input_size = 4
    hidden_size = 50
    num_layers = 5
    delta_t = 0.01
    n_steps = 90

    # Initialize and train the network
    net = DeepCrankNicolsonNetC(input_size, hidden_size, num_layers, delta_t, n_steps=n_steps)
    net.train(X0_train, Xn_train, X0_val, Xn_val, epochs=epochs)

    # Evaluate the model using the validation data
    X_outputs_forward_val, X_outputs_backward_val = evaluate(net, X0_val, Xn_val)
    return net, X_outputs_forward_val, X_outputs_backward_val

# Function to evaluate the model
def evaluate(net, X0_val, Xn_val):
    net.net.eval()
    with torch.no_grad():
        X_outputs_forward = net.forward(X0_val)
        X_outputs_backward = net.backward(Xn_val)
    return X_outputs_forward, X_outputs_backward

# Generating synthetic data
num_samples = 1000
input_size = 4
timesteps = np.linspace(0, 1000, num_samples)
data = np.array([np.sin(timesteps + np.random.uniform(-0.5, 0.5)) for _ in range(input_size)]).T
data += 0.1 * np.random.randn(num_samples, input_size)

# Splitting the data
train_split = int(0.7 * num_samples)
val_split = int(0.85 * num_samples)
X0_train = torch.tensor(data[:train_split], dtype=torch.float32)
Xn_train = torch.tensor(data[1:train_split+1], dtype=torch.float32)
X0_val = torch.tensor(data[train_split:val_split], dtype=torch.float32)
Xn_val = torch.tensor(data[train_split+1:val_split+1], dtype=torch.float32)

# Running the training function
trained_model, forward_val_results, backward_val_results = train_model(X0_train, Xn_train, X0_val, Xn_val, epochs=10)

# Results
print("Forward Validation Results Shape:", forward_val_results.shape)
print("Backward Validation Results Shape:", backward_val_results.shape)


def plot_time_evolution(forward_results, backward_results, original_data, selected_samples, selected_features):
    """
    Plot the time evolution of forward, backward, and original data for specific samples and features.

    Parameters:
        forward_results (numpy.ndarray): Forward validation results.
        backward_results (numpy.ndarray): Backward validation results.
        original_data (numpy.ndarray): Original data (test data).
        selected_samples (list): List of selected samples for visualization.
        selected_features (list): List of selected features for visualization.
    """
    backward_results_np = backward_results.cpu().numpy()  # Convert to NumPy array if it's a tensor
    forward_results_np = forward_results.cpu().numpy()  # Convert to NumPy array if it's a tensor


    for sample in selected_samples:
        for feature in selected_features:
            plt.figure(figsize=(12, 6))

            # Plotting the forward results
            plt.plot(forward_results_np[:, sample, feature], label='Forward Prediction', linestyle='-', marker='o')

            # Plotting the backward results
            plt.plot(backward_results_np[::-1, sample, feature], label='Backward Prediction', linestyle='--', marker='x')

            # Plotting the original data (test data)
            original_value = original_data[sample, feature]
            plt.plot([original_value] * (forward_results.shape[0]), label='Original Data', linestyle=':', marker='s')

            plt.title(f'Sample {sample}, Feature {feature} - Time Evolution')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.show()

def plot_prediction_error_histograms(X0, Xn, forward_results, backward_results):
    """
    Plot histograms of the forward and backward prediction errors.

    Parameters:
        X0 (numpy.ndarray): Original starting values.
        Xn (numpy.ndarray): Original ending values.
        forward_results (numpy.ndarray): Forward validation results.
        backward_results (numpy.ndarray): Backward validation results.
    """
    # Calculating the prediction errors
    forward_prediction_error = X0 - forward_results[0]
    backward_prediction_error = Xn - backward_results[-1]

    # Flattening the errors for histogram plotting
    forward_prediction_error_flattened = forward_prediction_error.flatten()
    backward_prediction_error_flattened = backward_prediction_error.flatten()

    # Plotting the histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Forward Prediction Error
    ax1.hist(forward_prediction_error_flattened, bins=50, edgecolor='black')
    ax1.set_title('Histogram of Forward Prediction Error (X0 - Forward Results)')
    ax1.set_xlabel('Error Value')
    ax1.set_ylabel('Frequency')

    # Backward Prediction Error
    ax2.hist(backward_prediction_error_flattened, bins=50, edgecolor='black')
    ax2.set_title('Histogram of Backward Prediction Error (Xn - Backward Results)')
    ax2.set_xlabel('Error Value')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Calling the function to plot the histograms using the obtained results
#plot_prediction_error_histograms(X0_val.cpu().numpy(), Xn_val.cpu().numpy(), forward_val_results.cpu().numpy(), backward_val_results.cpu().numpy())


# Selecting a few samples and features for visualization
selected_samples = [10, 50, 100] # Selecting specific samples
selected_features = [1, 2, 3] # Selecting specific features

# Example usage of the function with the obtained results and selected samples and features
#plot_time_evolution(forward_val_results, backward_val_results, data[train_split:val_split],
#                    selected_samples, selected_features)


def plot_single_sample_evolution(Xn_sample, forward_results_sample, backward_results_sample):
    """
    Plot the time evolution of a single sample for original Xn, forward prediction, and backward prediction.

    Parameters:
        Xn_sample (numpy.ndarray): Original Xn values for the selected sample.
        forward_results_sample (numpy.ndarray): Forward prediction results for the selected sample.
        backward_results_sample (numpy.ndarray): Backward prediction results for the selected sample.
    """
    plt.figure(figsize=(12, 6))

    # Plotting the forward results
    plt.plot(forward_results_sample, label='Forward Prediction', linestyle='-', marker='o')

    # Plotting the backward results
    plt.plot(backward_results_sample[::-1], label='Backward Prediction', linestyle='--', marker='x')

    # Plotting the original Xn values (all time steps)
    plt.plot(Xn_sample, label='Original Xn Data', linestyle=':', marker='s')

    plt.title('Time Evolution for Selected Sample')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


# Selecting a single sample index for visualization (e.g., the 10th sample)
selected_sample_index = 10

# Extracting the corresponding Xn values, forward results, and backward results for the selected sample
Xn_sample_selected = Xn_val.cpu().numpy()[selected_sample_index]
forward_results_sample_selected = forward_val_results[:, selected_sample_index]
backward_results_sample_selected = backward_val_results[:, selected_sample_index]

# Calling the function to plot the time evolution for the selected sample
plot_single_sample_evolution(Xn_sample_selected, forward_val_results.cpu().numpy(), backward_val_results.cpu().numpy())


