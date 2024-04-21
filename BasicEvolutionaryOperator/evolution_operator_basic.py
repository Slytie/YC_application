import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define a 3D grid
grid_size = 20
x = np.linspace(0, 10, grid_size)
y = np.linspace(0, 10, grid_size)
z = np.linspace(0, 10, grid_size)
X, Y, Z = np.meshgrid(x, y, z)
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# Number of components and samples
num_components = 4
num_samples_GMM = 5000

# Randomly define means for the Gaussian components in 3D space
means = np.random.rand(num_components, 3) * 10  # Values between 0 and 10 for each dimension

# Randomly define covariances for the Gaussian components
covariances = [np.diag(np.random.rand(3) + 1) for _ in range(num_components)]  # Diagonal covariance matrices

# Define a sparser transition matrix
sparse_transition_matrix = np.zeros((num_components, num_components))
sparse_transition_matrix[0, 2] = 0.8
sparse_transition_matrix[1, 3] = 0.7
sparse_transition_matrix[2, 1] = 0.6
sparse_transition_matrix[3, 0] = 0.5
row_sums = sparse_transition_matrix.sum(axis=1, keepdims=True)
sparse_transition_matrix /= row_sums + 1e-10

class EvolutionOperator(nn.Module):
    def __init__(self, size):
        super(EvolutionOperator, self).__init__()
        self.U = nn.Parameter(torch.eye(size))
    def forward(self, x):
        return torch.mm(x, self.U)

def sample_from_GMM(means, covariances, transition_matrix, num_samples):
    initial_samples = []
    evolved_samples = []
    for _ in range(num_samples):
        initial_component = np.random.choice(num_components)
        initial_sample = multivariate_normal.rvs(mean=means[initial_component], cov=covariances[initial_component])
        initial_samples.append(initial_sample)
        evolved_component = np.random.choice(num_components, p=transition_matrix[initial_component])
        evolved_sample = multivariate_normal.rvs(mean=means[evolved_component], cov=covariances[evolved_component])
        evolved_samples.append(evolved_sample)
    return np.array(initial_samples), np.array(evolved_samples)

# Generate dataset
initial_samples, evolved_samples = sample_from_GMM(means, covariances, sparse_transition_matrix, num_samples_GMM)

# Convert the 3D samples to torch tensors
initial_samples_tensor = torch.tensor(initial_samples, dtype=torch.float32)
evolved_samples_tensor = torch.tensor(evolved_samples, dtype=torch.float32)

# Model, loss, and optimizer for the GMM scenario
learning_rate = 0.01
num_epochs = 1500
model_GMM = EvolutionOperator(3)  # 3D space
criterion = nn.MSELoss()
optimizer = optim.Adam(model_GMM.parameters(), lr=learning_rate)

# Train the model for the GMM scenario
losses_GMM = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model_GMM(initial_samples_tensor)
    loss = criterion(outputs, evolved_samples_tensor)
    loss.backward()
    optimizer.step()
    losses_GMM.append(loss.item())

# Compute the ground truth
ground_truth = np.zeros((grid_size, grid_size, grid_size))
for i in range(num_components):
    for j in range(num_components):
        transition_prob = sparse_transition_matrix[i, j]
        density = multivariate_normal(mean=means[i], cov=covariances[i]).pdf(positions)
        ground_truth += transition_prob * density.reshape(grid_size, grid_size, grid_size)

# Predicted Density over the grid
predicted_states_on_grid = model_GMM(torch.tensor(positions, dtype=torch.float32)).detach().numpy()
predicted_density = np.linalg.norm(predicted_states_on_grid - positions, axis=1).reshape(grid_size, grid_size, grid_size)

# Plotting both the ground truth and predicted density slices side by side
slice_idx = grid_size // 2
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
c1 = axs[0].imshow(ground_truth[:, :, slice_idx], extent=(0, 10, 0, 10))
fig.colorbar(c1, ax=axs[0], fraction=0.046, pad=0.04)
axs[0].set_title(f'Ground Truth Density Slice at Z = {z[slice_idx]:.2f}')
c2 = axs[1].imshow(predicted_density[:, :, slice_idx], extent=(0, 10, 0, 10))
fig.colorbar(c2, ax=axs[1], fraction=0.046, pad=0.04)
axs[1].set_title(f'Predicted Density Slice at Z = {z[slice_idx]:.2f}')
plt.show()
