import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.infer import SVI, Trace_ELBO

#the loss between the predicted and true outputs should be the KL divergence across the same kth dimension as the softmax


# Bayesian Convolutional Layer with Sparse Bayesian Learning
class SparseBayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SparseBayesianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Define priors for weights and biases
        self.weight_prior = dist.Normal(0, 1)
        self.bias_prior = dist.Normal(0, 1)

        # Define parameters (these will store our posterior beliefs)
        self.weight_mu = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_rho = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_rho = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Sample weights and biases from their posteriors
        weight = self.weight_prior.sample(self.weight_mu.shape) * torch.sigmoid(self.weight_rho) + self.weight_mu
        bias = self.bias_prior.sample(self.bias_mu.shape) * torch.sigmoid(self.bias_rho) + self.bias_mu

        return nn.functional.conv2d(x, weight, bias)


# Bayesian Fully Connected Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Define priors for weights and biases
        self.weight_prior = dist.Normal(0, 1)
        self.bias_prior = dist.Normal(0, 1)

        # Define parameters (these will store our posterior beliefs)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Sample weights and biases from their posteriors
        weight = self.weight_prior.sample(self.weight_mu.shape) * torch.sigmoid(self.weight_rho) + self.weight_mu
        bias = self.bias_prior.sample(self.bias_mu.shape) * torch.sigmoid(self.bias_rho) + self.bias_mu

        return nn.functional.linear(x, weight, bias)


# Bayesian Decoding Layer
class BayesianDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BayesianDeconv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define priors for weights and biases
        self.weight_prior = dist.Normal(0, 1)
        self.bias_prior = dist.Normal(0, 1)

        # Define parameters (these will store our posterior beliefs)
        self.weight_mu = nn.Parameter(torch.zeros(in_channels, out_channels, kernel_size, kernel_size))
        self.weight_rho = nn.Parameter(torch.zeros(in_channels, out_channels, kernel_size, kernel_size))
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_rho = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Sample weights and biases from their posteriors
        weight = self.weight_prior.sample(self.weight_mu.shape) * torch.sigmoid(self.weight_rho) + self.weight_mu
        bias = self.bias_prior.sample(self.bias_mu.shape) * torch.sigmoid(self.bias_rho) + self.bias_mu

        return nn.functional.conv_transpose2d(x, weight, bias)


class StochasticProcessNeuralModel(nn.Module):
    def __init__(self, in_channels, kernel_size, N, K):
        super(StochasticProcessNeuralModel, self).__init__()

        self.N = N
        self.K = K
        self.kernel_size = kernel_size

        # Encoding layers
        self.encoder = nn.Sequential(
            SparseBayesianConv2d(in_channels, 32, kernel_size),
            nn.ReLU(),
            SparseBayesianConv2d(32, 64, kernel_size),
            nn.ReLU()
        )

        # Latent representation layer
        flattened_dim = 64 * (self.N - 2 * (self.kernel_size - 1)) * (self.K - 2 * (self.kernel_size - 1))
        self.latent_layer = BayesianLinear(flattened_dim, flattened_dim)

        # Decoding layers
        self.decoder = nn.Sequential(
            BayesianDeconv2d(64, 32, kernel_size),
            nn.ReLU(),
            BayesianDeconv2d(32, in_channels, kernel_size)
        )

    def forward(self, x):
        # Encoding phase
        x_encoded = self.encoder(x)
        print(f"Shape after encoding: {x_encoded.shape}")

        # Flatten the tensor for the fully connected layer
        x_flattened = x_encoded.view(x_encoded.size(0), -1)
        print(f"Shape after flattening: {x_flattened.shape}")

        # Latent representation
        r = self.latent_layer(x_flattened)
        print(f"Shape of latent representation: {r.shape}")

        # Decoding phase
        x_reshaped = r.view(r.size(0), 64, self.N - 2 * (self.kernel_size - 1), self.K - 2 * (self.kernel_size - 1))
        print(f"Shape after reshaping for decoding: {x_reshaped.shape}")
        x_decoded = self.decoder(x_reshaped)
        print(f"Shape after decoding: {x_decoded.shape}")

        # Apply softmax along the K dimension
        x_final = F.softmax(x_decoded, dim=3)

        return x_final


# Shape Tests
def test_shapes():
    N = 10
    K = 20

    # Dummy input data
    dummy_data = torch.randn(5, 1, N, K)  # Batch size of 5

    model = StochasticProcessNeuralModel(1, 3, N, K)

    # Encoding phase
    encoded = model.encoder(dummy_data)
    assert encoded.shape == (5, 64, N - 2 * (3 - 1), K - 2 * (
                3 - 1)), f"Expected {(5, 64, N - 2 * (3 - 1), K - 2 * (3 - 1))}, got {encoded.shape}"

    # Decoding phase
    decoded = model(dummy_data)
    assert decoded.shape == dummy_data.shape, f"Expected {dummy_data.shape}, got {decoded.shape}"

    print("All shape tests passed!")


test_shapes()

def test_softmax_output():
    N = 10
    K = 20

    # Dummy input data
    dummy_data = torch.randn(5, 1, N, K)  # Batch size of 5
    model = StochasticProcessNeuralModel(1, 3, N, K)
    output = model(dummy_data)

    # Check values are between 0 and 1
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output values are not between 0 and 1."

    # Check sum across the K dimension is 1
    sum_across_dims = torch.sum(output, dim=3)
    assert torch.allclose(sum_across_dims, torch.ones_like(sum_across_dims), atol=1e-5), "Sum across dimensions is not close to 1."



def test_bayesian_sampling():
    N = 10
    K = 20

    # Dummy input data
    dummy_data = torch.randn(5, 1, N, K)  # Batch size of 5
    model = StochasticProcessNeuralModel(1, 3, N, K)
    outputs = [model(dummy_data) for _ in range(5)]

    # Check if multiple outputs are different, implying sampling from distributions
    for i in range(4):
        assert not torch.equal(outputs[i], outputs[
            i + 1]), f"Outputs {i} and {i + 1} are identical. Bayesian sampling might not be working."

    print("Bayesian sampling test passed!")


