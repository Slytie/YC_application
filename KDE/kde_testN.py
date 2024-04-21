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


def generate_discrete_exponential(lmbda=0.5, K=10):
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


class DifferentiableKDEMahalanobis(nn.Module):
    def __init__(self, dim, init_cov_matrix=None):
        super(DifferentiableKDEMahalanobis, self).__init__()

        # Initialize bandwidth matrix
        self.H_bandwidth = nn.Parameter(torch.full((dim, dim), 0.05) + torch.eye(dim) * 1)

        # For Mahalanobis distance
        if init_cov_matrix is None:
            init_cov_matrix = torch.eye(dim)
        L = torch.linalg.cholesky(init_cov_matrix)
        self.L = nn.Parameter(L)

    @property
    def cov_matrix(self):
        return torch.mm(self.L, self.L.t())

    @property
    def cov_inv(self):
        return torch.inverse(self.cov_matrix)

    def _gaussian_kernel1(self, z):
        normalization = torch.sqrt(
            torch.det(self.H_bandwidth) * (2.0 * torch.tensor(np.pi)) ** self.H_bandwidth.size(0))
        return torch.exp(-0.5 * z) / normalization

    def _gaussian_kernel(self, z):
        return (1.0 / (1.0 * torch.sqrt(2.0 * torch.tensor(np.pi)))) * torch.exp(
            -0.5 * (z / 1.0) ** 2)

    def _mahalanobis_distance(self, shape, query):
        grids = torch.meshgrid(*[torch.arange(k) for k in shape])
        stacked = torch.stack(grids, dim=0).float()

        query_tensor = torch.tensor(query).float()
        for _ in range(len(shape)):
            query_tensor = query_tensor.unsqueeze(-1)
        diff = stacked - query_tensor

        distance_sq = torch.einsum('...i,ij,...j', diff.reshape(-1, diff.shape[0]), self.cov_inv,
                                   diff.reshape(-1, diff.shape[0]))
        distance_sq = distance_sq.reshape(shape)

        return torch.sqrt(distance_sq)

    def forward(self, sample_distributions):
        combined_kde_values = torch.zeros_like(sample_distributions[0])
        for space_probs in sample_distributions:
            kde_values = torch.zeros_like(space_probs)
            for idx in np.ndindex(space_probs.shape):
                distances = self._mahalanobis_distance(space_probs.shape, idx)
                kernel_weights = self._gaussian_kernel(distances)
                kde_values[idx] = torch.sum(kernel_weights * space_probs)
            kde_values /= kde_values.sum()
            combined_kde_values += kde_values
        combined_kde_values /= len(sample_distributions)
        return kde_values

# Define parameters
means = torch.tensor([1, 1, 1])
std_devs = torch.tensor([0.7, 0.7, 0.7])
K = 10
N = 3

cov = torch.tensor([
    [0.7, 0.05, 0.05],
    [0.05, 0.7, 0.05],
    [0.05, 0.05, 0.7]
])

# Generate N independent distributions over K locations
distributions1 = torch.stack([generate_discrete_normal(mean, std_dev, K=10) + generate_discrete_exponential() for mean, std_dev in zip(means, std_devs)])
distributions2 = torch.stack([generate_discrete_normal(mean, std_dev, K=10) for mean, std_dev in zip(means, std_devs)])
distributions3 = torch.stack([generate_discrete_exponential() for mean, std_dev in zip(means, std_devs)])

d1 = tensor_product_of_distributions(distributions3)
d2 = tensor_product_of_distributions(distributions3)
d3 = tensor_product_of_distributions(distributions3)

distributions_3d = [d1, d2, d3]

# Generate 3D sample distributions of shape 10x10x10
#distributions_3d = [torch.rand((10, 10, 10)) for _ in range(3)]


# Instantiate the 3D model
model_3d = DifferentiableKDEMahalanobis(dim=3, init_cov_matrix=cov)

# Compute the KDE output for these distributions
kde_output_3d = model_3d(distributions_3d)


def visualize_slices(sample_distributions, kde_output):
    fig, axs = plt.subplots(len(sample_distributions) + 1, 1, figsize=(8, 20))

    for i, dist in enumerate(sample_distributions):
        sns.heatmap(dist[3].numpy(), ax=axs[i], cmap='viridis', cbar=True)
        axs[i].set_title(f'Sample Distribution {i + 1} (Slice 5)')
        axs[i].axis('off')

    # Plotting KDE output using detach().numpy()
    sns.heatmap(kde_output[3].detach().numpy(), ax=axs[-1], cmap='viridis', cbar=True)
    axs[-1].set_title('KDE Output (Slice 5)')
    axs[-1].axis('off')

    plt.tight_layout()
    plt.show()


# Visualize the slices
visualize_slices(distributions_3d, kde_output_3d)
