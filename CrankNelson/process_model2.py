import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np


class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DeepNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        for _ in range(self.num_layers - 2):
            out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class DeepEulerNet:
    def __init__(self, input_size, hidden_size, num_layers, delta_t, n_steps, newton_iterations=10, lr=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DeepNet(input_size, hidden_size, num_layers).to(self.device)
        self.delta_t = delta_t
        self.n_steps = n_steps
        self.newton_iterations = newton_iterations
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, X0):
        X = X0
        X_outputs = [X0]
        for _ in range(self.n_steps):
            X = X + self.delta_t * self.net(X)
            X_outputs.append(X)
        return torch.stack(X_outputs)

    def forward(self, X0):
        X = X0
        X_outputs = [X0]
        for _ in range(self.n_steps):
            k1 = self.net(X)
            k2 = self.net(X + 0.5 * self.delta_t * k1)
            k3 = self.net(X + 0.5 * self.delta_t * k2)
            k4 = self.net(X + self.delta_t * k3)
            X = X + (self.delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            X_outputs.append(X)
        return torch.stack(X_outputs)

    def backward(self, Xn):
        X = Xn
        X_outputs = [Xn]
        for _ in range(self.n_steps):
            X_prev = X
            for _ in range(self.newton_iterations):  # Apply Newton's method for each time step
                f_X = self.net(X)
                X = X_prev - self.delta_t * f_X
            X_outputs.append(X)
        return torch.stack(X_outputs[::-1])

    def compute_loss(self, X0, Xn):
        X_outputs_forward = self.forward(X0)
        X_outputs_backward = self.backward(Xn)
        # Compute loss at each timestep
        losses = [self.criterion(X_outputs_forward[i], Xn[i]) + 0* self.criterion(X_outputs_backward[i], X0[i]) for i in
                  range(self.n_steps)]
        return sum(losses)

    def train(self, X0_train, Xn_train, X0_val, Xn_val, epochs=1000):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Compute loss on train data
            loss_train = self.compute_loss(X0_train, Xn_train)
            train_losses.append(loss_train.item())

            # Backpropagate gradients and update parameters
            loss_train.backward()
            self.optimizer.step()

            # Compute loss on validation data
            with torch.no_grad():
                loss_val = self.compute_loss(X0_val, Xn_val)
                val_losses.append(loss_val.item())

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {loss_train.item()}, Validation Loss: {loss_val.item()}')

        return train_losses, val_losses


def plot_results_forward_backward_single_variable(X0_test, Xn_test, X_outputs_forward, X_outputs_backward):
    n_variables = X0_test.shape[1]
    fig, axs = plt.subplots(n_variables, 1, figsize=(12, 6 * n_variables))

    for var in range(n_variables):
        # Plot the forward prediction
        forward_values = X_outputs_forward[:, 0, var] if X_outputs_forward.dim() == 3 else X_outputs_forward[:, var]
        axs[var].plot(forward_values.cpu().detach().numpy(), label='Forward Prediction')

        backward_values = X_outputs_backward[:, 0, var] if X_outputs_backward.dim() == 3 else X_outputs_backward[:, var]
        #axs[var].plot(backward_values.cpu().detach().numpy().squeeze(), label='Backward Prediction')

        # Plot the test data
        axs[var].plot(Xn_test[0, :, var], 'ro-', label='Test data')

        axs[var].set_title(f'Variable {var + 1}')
        axs[var].set_xlabel('Time step')
        axs[var].set_ylabel('Value')
        axs[var].legend()

    plt.tight_layout()
    plt.show()



class ODEUpdated:
    def __init__(self, n_variables=5, A=None, delta_t=1):
        self.n_variables = n_variables
        self.A = A if A is not None else np.random.rand(self.n_variables, self.n_variables)
        self.delta_t = delta_t

    def generate_data(self, batch_size=100, n_steps=6):
        X0 = np.random.rand(batch_size, self.n_variables)
        X = X0.copy()
        X_outputs = [X0]
        for _ in range(n_steps):  # Generate n_steps more steps
            X = X + self.delta_t * np.dot(X, self.A.T)  # Forward Euler step
            X_outputs.append(X)
        return torch.tensor(X0, dtype=torch.float32), torch.tensor(np.array(X_outputs),
                                                                   dtype=torch.float32)  # Convert list to numpy array before converting to tensor


if __name__ == "__main__":
    # Initialize the ODE model
    ode_model = ODEUpdated(n_variables=5, delta_t=1)

    # Generate some data
    X0_train, Xn_train = ode_model.generate_data(batch_size=100, n_steps=6)
    X0_val, Xn_val = ode_model.generate_data(batch_size=20, n_steps=6)

    # Initialize the DeepEulerNet model
    deep_euler_net = DeepEulerNet(input_size=5, hidden_size=30, num_layers=10, delta_t=1, n_steps=6, newton_iterations=10, lr=0.0005)

    # Train the model
    train_losses, val_losses = deep_euler_net.train(X0_train, Xn_train, X0_val, Xn_val, epochs=1000)

    # Generate some test data
    X0_test, Xn_test = ode_model.generate_data(batch_size=1, n_steps=6)

    # Make predictions using the trained model
    X_outputs_forward = deep_euler_net.forward(X0_test.to(deep_euler_net.device))
    X_outputs_backward = deep_euler_net.backward(Xn_test.to(deep_euler_net.device))

    # Plot results
    plot_results_forward_backward_single_variable(X0_test, Xn_test, X_outputs_forward, X_outputs_backward)

    # You can plot the loss over time to see how your model is improving
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()


