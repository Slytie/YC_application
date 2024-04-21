import torch
import numpy as np
from torch import nn, optim
import pymunk

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
    assert (fa * fb < 0).flatten().all(), "The function must have different signs at a and b."

    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        mask = fa * fc < 0
        a = torch.where(mask, a, c)
        b = torch.where(mask, c, b)
        fa = torch.where(mask, fa, fc)
        fb = torch.where(mask, fc, fb)

        # Check the tolerance
        if torch.max(torch.abs(b - a)) < tol:
            break

    return (a + b) / 2

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

        loss_forward = self.criterion(X_outputs_forward, X0_seq)
        loss_backward = self.criterion(X_outputs_backward, Xn_seq)

        reg_term = self.reg_factor * sum(p.norm() for p in self.net.parameters())

        return loss_forward + loss_backward + reg_term

    def train(self, X0_seq, Xn_seq, X0_val_seq, Xn_val_seq, epochs=1000):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.compute_loss(X0_seq, Xn_seq)
            loss.backward()
            self.optimizer.step()

# Function to generate particle box data with intermediates
def generate_particle_box_data_with_intermediates(N, X, Z, num_simulations=100, num_steps=10):
    X_data_all = np.zeros((num_simulations, num_steps+1, N, 4))
    for sim in range(num_simulations):
        space = pymunk.Space()
        space.gravity = (0.0, -9.8)
        particles = []
        for _ in range(N):
            mass = 1.0
            radius = X / Z
            moment = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, moment)
            body.position = tuple(np.random.uniform(0, X, size=2))
            body.velocity = tuple(np.random.uniform(-1, 1, size=2))
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 1.0
            space.add(body, shape)
            particles.append(body)
        for i in range(4):
            a = (i % 2) * X
            b = (i // 2) * X
            boundary = pymunk.Segment(space.static_body, (a, b), (X - a, X - b), 1.0)
            boundary.elasticity = 1.0
            space.add(boundary)
        for j, particle in enumerate(particles):
            X_data_all[sim, 0, j, :2] = particle.position
            X_data_all[sim, 0, j, 2:] = particle.velocity
        for step in range(num_steps):
            space.step(1.0 / 60.0)
            for j, particle in enumerate(particles):
                X_data_all[sim, step+1, j, :2] = particle.position
                X_data_all[sim, step+1, j, 2:] = particle.velocity

    # Reshape and convert to PyTorch tensors
    X_data_all = torch.from_numpy(X_data_all.reshape(num_simulations, num_steps+1, N * 4)).float()
    return X_data_all

# Generate training and validation data
N = 5
X = 10
Z = 1
num_simulations = 100
num_steps = 10
X0_train = generate_particle_box_data_with_intermediates(N, X, Z, num_simulations, num_steps)
Xn_train = generate_particle_box_data_with_intermediates(N, X, Z, num_simulations, num_steps)
X0_val = generate_particle_box_data_with_intermediates(N, X, Z, num_simulations//10, num_steps)
Xn_val = generate_particle_box_data_with_intermediates(N, X, Z, num_simulations//10, num_steps)

# Initialize and train the network
input_size = N * 4
hidden_size = 50
num_layers = 5
delta_t = 0.01
net = DeepCrankNicolsonNetC(input_size, hidden_size, num_layers, delta_t, num_steps)
net.train(X0_train, X0_train, X0_val, Xn_val, epochs=1000)

import matplotlib.pyplot as plt
# Function to evaluate the model
def evaluate(net, X0_val, Xn_val):
    net.net.eval()  # Set the network in evaluation mode
    with torch.no_grad():
        X_outputs_forward = net.forward(X0_val)
        X_outputs_backward = net.backward(Xn_val)
    return X_outputs_forward, X_outputs_backward

# Function to plot the forward and backward pass results
def plot_results(X0, Xn, X_outputs_forward, X_outputs_backward, example_idx=0):
    plt.figure(figsize=(12, 6))

    # Plotting the forward pass
    plt.subplot(2, 1, 1)
    plt.plot(X0[example_idx].numpy(), label='Actual')
    plt.plot(X_outputs_forward[example_idx].numpy(), label='Predicted')
    plt.title('Forward Pass')
    plt.legend()

    # Plotting the backward pass
    plt.subplot(2, 1, 2)
    plt.plot(Xn[example_idx].numpy()[::-1], label='Actual')
    plt.plot(X_outputs_backward[example_idx].numpy()[::-1], label='Predicted')
    plt.title('Backward Pass')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Evaluate the model using the validation data
X_outputs_forward_val, X_outputs_backward_val = evaluate(net, X0_val, Xn_val)

# Plot the results for a specific example
plot_results(X0_val, Xn_val, X_outputs_forward_val, X_outputs_backward_val, example_idx=0)