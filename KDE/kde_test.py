import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

def generate_discrete_normal(mean, std_dev, K=8):
    """Generate a discrete normal distribution over K locations."""
    locations = torch.linspace(0, K - 1, K)
    probs = torch.exp(-0.5 * ((locations - mean) / std_dev) ** 2)
    probs /= probs.sum()
    return probs


def generate_discrete_exponential(lmbda=0.5, K=8):
    """Generate a discrete exponential distribution over K locations."""
    locations = torch.linspace(0, K - 1, K)
    probs = lmbda * torch.exp(-lmbda * locations)
    probs /= probs.sum()
    return probs

def generate_discrete_gamma(shape, scale, K=8):
    """Generate a discrete gamma distribution over K locations."""
    locations = torch.linspace(0, K - 1, K)
    probs = (locations**(shape - 1) * torch.exp(-locations / scale)) / (torch.exp(torch.lgamma(torch.tensor([shape]))) * scale**shape)
    probs /= probs.sum()
    return probs

def generate_discrete_uniform(K=8):
    """Generate a discrete uniform distribution over K locations."""
    return torch.ones(K) / K

def tensor_product_of_distributions(distributions):
    """Compute the tensor product for an arbitrary number of distributions."""

    # Number of distributions
    N = distributions.shape[0]

    # Generate the einsum input string for each distribution
    # For N=3, this will generate ['i', 'j', 'k'] and so on for other N
    input_strings = [chr(97 + i) for i in range(N)]

    # Construct the full einsum string
    einsum_string = ','.join(input_strings) + '->' + ''.join(input_strings)

    # Compute the tensor product using einsum
    return torch.einsum(einsum_string, *[distributions[i] for i in range(N)])


class KDE(nn.Module):
    def __init__(self, bandwidth=1.0, dim=3, cov_matrix=None):
        super(KDE, self).__init__()
        self.bandwidth = bandwidth
        self.dim = dim
        self.cov_matrix = cov_matrix
        self.cov_inv = torch.inverse(self.cov_matrix)

    def forward(self, space_probs):
        kde_values = torch.zeros_like(space_probs)
        for idx in np.ndindex(space_probs.shape):
            distances = self._mahalanobis_distance(space_probs.shape, idx)
            kernel_weights = self._gaussian_kernel(distances)
            kde_values[idx] = torch.sum(kernel_weights * space_probs)
        kde_values /= kde_values.sum()
        return kde_values

    def _gaussian_kernel(self, z):
        return (1.0 / (self.bandwidth * torch.sqrt(2.0 * torch.tensor(np.pi)))) * torch.exp(-0.5 * (z / self.bandwidth) ** 2)

    def _mahalanobis_distance(self, shape, query):
        grids = torch.meshgrid(*[torch.arange(k) for k in shape])
        stacked = torch.stack(grids, dim=0).float()
        diff = stacked - torch.tensor(query).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        mahalanobis_sq = torch.einsum("i...,ij,j...", diff, self.cov_inv, diff)
        return torch.sqrt(mahalanobis_sq)



import matplotlib.pyplot as plt
import seaborn as sns

# Visualization function
def plot_heatmap(data, title, ax):
    sns.heatmap(data, cmap='viridis', ax=ax)
    ax.set_title(title)
    ax.axis('off')


def combine_distributions(distributions):
    """Combine given distributions by adding and then normalizing."""
    combined = sum(distributions)
    return combined / combined.sum()

# Define the space and generate a 3D discrete normal distribution with reduced variance
mean = [4, 4, 4]
cov = torch.tensor([
    [0.7, 0.05, 0.05],
    [0.05, 0.7, 0.05],
    [0.05, 0.05, 0.7]
])

# Define parameters
means = torch.tensor([4, 1, 0])
std_devs = torch.tensor([0.7, 0.7, 0.7])
K = 8
N = 3

probs_e = generate_discrete_exponential()

# Generate N independent distributions over K locations
distributions = torch.stack([generate_discrete_normal(mean, std_dev, K) + generate_discrete_exponential() for mean, std_dev in zip(means, std_devs)])
distributions = torch.stack([generate_discrete_exponential() for mean, std_dev in zip(means, std_devs)])
probs_t = tensor_product_of_distributions(distributions)

# Instantiate the differentiable KDE class with Mahalanobis distance and compute KDE values
kde_mahalanobis = KDE(bandwidth=1.0, dim=3, cov_matrix=cov)
kde_values_mahalanobis = kde_mahalanobis(probs_t)

# Check if the results are valid (sums up to 1)
kde_values_mahalanobis.sum()

# Create a 2x2 grid of plots for visualization using the Mahalanobis-based KDE values
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot the original probabilities and Mahalanobis-based KDE values for the slice (dimension 2 vs. dimension 3)
plot_heatmap(probs_t.numpy()[4, :, :], "Original Probabilities (Dim 2 vs. Dim 3)", axes[0, 0])
plot_heatmap(kde_values_mahalanobis.detach().numpy()[4, :, :], "Mahalanobis KDE Values (Dim 2 vs. Dim 3)", axes[0, 1])

# Revisiting the slices for dimension 1 vs. dimension 2 but using the middle slice along dimension 3
plot_heatmap(probs_t.numpy()[:, :, 4], "Original Probabilities (Dim 1 vs. Dim 2)", axes[1, 0])
plot_heatmap(kde_values_mahalanobis.detach().numpy()[:, :, 4], "Mahalanobis KDE Values (Dim 1 vs. Dim 2)", axes[1, 1])

plt.tight_layout()
plt.show()



