from torch import nn, optim
import numpy as np
import torch


def plot_image_comparison(true_tensor, pred_tensor, coordinates, image_shape):
    # Unflatten and detach the tensors, then convert to numpy arrays for plotting
    true_image = true_tensor.detach().cpu().numpy().reshape(image_shape)
    pred_image = pred_tensor.detach().cpu().numpy().reshape(image_shape)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(true_image, cmap='gray')
    axs[0].set_title('True Image')
    axs[0].set_xlabel(f'Coordinates: {coordinates}')

    axs[1].imshow(pred_image, cmap='gray')
    axs[1].set_title('Predicted Image')
    axs[1].set_xlabel(f'Coordinates: {coordinates}')

    plt.show()

def plot_results_forward_backward_single_variable(X0_test, Xn_test, X_outputs_forward, X_outputs_backward, time_steps):
    n_variables = X0_test.shape[1]
    fig, axs = plt.subplots(n_variables, 1, figsize=(12, 6 * n_variables))

    for var in range(n_variables):
        # Plot the forward prediction
        axs[var].plot(range(time_steps + 1), X_outputs_forward[:, 0, var].cpu().detach().numpy(), label='Forward Prediction')

        # Plot the backward prediction
        #axs[var].plot(range(time_steps + 1), X_outputs_backward[:, 0, var].cpu().detach().numpy()[::-1], label='Backward Prediction')

        # Plot the test data
        axs[var].plot([0, time_steps], [X0_test[0, var], Xn_test[0, var]], 'ro-', label='Test data')

        axs[var].set_title(f'Variable {var + 1}')
        axs[var].set_xlabel('Time step')
        axs[var].set_ylabel('Value')
        axs[var].legend()

    plt.tight_layout()
    plt.show()


def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Implements the bisection root-finding method in PyTorch.

    Args:
    f: A PyTorch function for which we want to find a root.
    a, b: The interval in which to search for a root.
    tol: The desired precision.
    max_iter: Maximum number of iterations to perform.

    Returns:
    The root of the function f in the interval [a, b].
    """
    fa = f(a)
    fb = f(b)
    assert (fa * fb).item() < 0, "The function must have different signs at a and b."

    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        # Update the interval
        a, b = torch.where(fa * fc < 0, (a, c), (c, b))
        fa, fb = torch.where(fa * fc < 0, (fa, fc), (fc, fb))

        # Check the tolerance
        if torch.abs(b - a).item() < tol:
            break

    return (a + b) / 2


class DeepCrankNicolsonNet:
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
            f_X = self.net(X)
            X_new = X + 0.5 * self.delta_t * (f_X + self.net(X + self.delta_t * f_X))
            X = X_new
            X_outputs.append(X)
        return torch.stack(X_outputs)

    def backward(self, Xn):
        X = Xn
        X_outputs = [Xn]
        for _ in range(self.n_steps):
            X_prev = X
            for _ in range(self.newton_iterations):  # Apply Newton's method for each time step
                f_X = self.net(X)
                X = X_prev - 0.5 * self.delta_t * (f_X + self.net(X - self.delta_t * f_X))
            X_outputs.append(X)
        return torch.stack(X_outputs[::-1])

    def compute_loss(self, X0, Xn):
        X_outputs_forward = self.forward(X0)
        X_outputs_backward = self.backward(Xn)

        # Loss for first and last values for forward and backward predictions
        loss_first_last_forward = self.criterion(X_outputs_forward[0], X0) +  self.criterion(X_outputs_forward[-1], Xn)
        loss_first_last_backward = self.criterion(X_outputs_backward[0], Xn) + self.criterion(X_outputs_backward[-1], X0)

        # Reverse the tensor and perform the slicing
        X_outputs_backward_reversed = torch.flip(X_outputs_backward, [0])

        # Loss for the difference between the backward and forward predictions for the intermediate steps
        loss_intermediate = self.criterion(X_outputs_forward[1:-1], X_outputs_backward_reversed[1:-1])

        # Add a regularization term
        reg_term = self.reg_factor * sum(p.norm()**2 for p in self.net.parameters())

        return 1 * loss_first_last_forward + 0 *loss_first_last_backward + 0 * loss_intermediate + 1 *reg_term

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
            for _ in range(self.newton_iterations):  # Apply Newton's method for each time step
                f_X = self.net(X)
                X = bisection(
                    f=lambda x: x - X_prev - self.delta_t * 0.5 * (self.net(X_prev) + self.net(x)),
                    a=X_prev - 10, b=X_prev + 10, tol=1e-6, max_iter=100
                )
            X_outputs.append(X)
        return torch.stack(X_outputs[::-1])

    def backward(self, Xn):
        X = Xn
        X_outputs = [Xn]
        for _ in range(self.n_steps):
            X_prev = X
            for _ in range(self.newton_iterations):  # Apply Newton-Raphson method for each time step
                f_X = self.net(X)
                df_X = torch.autograd.grad(f_X.sum(), X, create_graph=True)[0]
                X = X - (f_X + self.delta_t * df_X - X_prev) / (df_X + self.delta_t)
            X_outputs.append(X)
        return torch.stack(X_outputs[::-1])

    def compute_loss(self, X0, Xn):
        X_outputs_forward = self.forward(X0)
        X_outputs_backward = self.backward(Xn)

        reg_term = self.reg_factor * sum(p.norm() for p in self.net.parameters())

        loss = self.criterion(X_outputs_forward, X0) + self.criterion(X_outputs_backward, Xn) + reg_term
        return loss

    def train(self, X0_train, Xn_train, epochs=1000):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.compute_loss(X0_train, Xn_train)
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
        return loss.item()


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

        # Loss for first and last values for forward and backward predictions
        loss_first_last_forward = self.criterion(X_outputs_forward[0], X0) + 4 * self.criterion(X_outputs_forward[-1],
                                                                                                Xn)
        loss_first_last_backward = self.criterion(X_outputs_backward[0], Xn) + self.criterion(X_outputs_backward[-1],
                                                                                              X0)

        # Reverse the tensor and perform the slicing
        X_outputs_backward_reversed = torch.flip(X_outputs_backward, [0])

        # Loss for the difference between the backward and forward predictions for the intermediate steps
        loss_intermediate = self.criterion(X_outputs_forward[1:-1], X_outputs_backward_reversed[1:-1])

        # Add a regularization term
        reg_term = 0.01 * sum(p.norm() for p in self.net.parameters())

        return 1 * loss_first_last_forward + 0 * loss_first_last_backward + 0 * loss_intermediate + reg_term


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
