import pyro
import torch
import pyro.distributions as dist
from torch import nn
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader


class BayesianStochasticProcessModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_components):
        super(BayesianStochasticProcessModel, self).__init__()

        # Define architecture dimensions
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_components = output_components

        # Define Bayesian layers - As an example, we'll use Gaussian distributions for weights
        self.layers = []
        all_dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            self.layers.append(self.bayesian_layer(in_dim, out_dim))

        # Define output layer for Dirichlet Mixture Model
        # We'll need nodes for mixture weights and nodes for each mixture component's parameters
        output_nodes = self.output_components + self.output_components * 2 * self.input_dim
        self.output_layer = self.bayesian_layer(all_dims[-1], output_nodes)

    def bayesian_layer(self, in_dim, out_dim):
        """Defines a Bayesian layer with Gaussian weights."""
        # Using Xavier initialization for the mean
        mean = nn.Parameter(torch.randn(out_dim, in_dim) * torch.sqrt(torch.tensor(2.0) / (out_dim + in_dim)))
        log_std = nn.Parameter(torch.zeros(out_dim, in_dim))
        return mean, log_std

    def forward(self, x, y=None):
        for i, (mean, log_std) in enumerate(self.layers):
            weight = mean + torch.exp(log_std) * torch.randn_like(mean)
            x = F.softplus(torch.mm(x, weight.t()))  # Using softplus activation

        # Output layer for Dirichlet Mixture Model parameters
        mean, log_std = self.output_layer
        weight = mean + torch.exp(log_std) * torch.randn_like(mean)
        output = torch.mm(x, weight.t())  # Explicit matrix multiplication

        # Ensure the output tensor has the expected shape
        output = output.reshape(x.size(0), -1)  # Reshape to (batch_size, -1)

        # Extracting mixture weights and component parameters
        mixture_weights = F.softmax(output[:, :self.output_components], dim=1)
        component_params = output[:, self.output_components:]

        return mixture_weights, component_params

    def guide(self, x,y=None):
        # Similar to forward but with 'pyro.sample' statements with the 'guide' suffix
        for i, (mean, log_std) in enumerate(self.layers):
            weight = pyro.sample(f"weight_{i}_guide", dist.Normal(mean, torch.exp(log_std)).to_event(2))
            x = F.softplus(torch.mm(x, weight.t()))

        # Output layer for the Dirichlet Mixture Model
        mean, log_std = self.output_layer
        output_weight = pyro.sample("output_weight_guide", dist.Normal(mean, torch.exp(log_std)).to_event(2))
        output = torch.mm(x, output_weight.t())

        return output

    def train(self, data_loader, epochs=10, lr=0.001):
        # Define the loss and optimizer
        svi = pyro.infer.SVI(self, self.guide, pyro.optim.Adam({"lr": lr}), loss=pyro.infer.Trace_ELBO())

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for inputs, outputs in data_loader:
                loss = svi.step(inputs, outputs)
                total_loss += loss
            avg_loss = total_loss / len(data_loader.dataset)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        return avg_loss


import unittest
import torch


class TestBayesianStochasticProcessModel(unittest.TestCase):

    def setUp(self):
        # Initialize a model for testing
        self.input_dim = 5
        self.hidden_dims = [10, 8]
        self.output_components = 3
        self.model = BayesianStochasticProcessModel(self.input_dim, self.hidden_dims, self.output_components)

    def test_model_initialization(self):
        """Test that the model initializes without errors."""
        self.assertIsInstance(self.model, BayesianStochasticProcessModel)
        self.assertEqual(len(self.model.layers), len(self.hidden_dims))

    def test_forward_pass(self):
        """Test forward pass of the model."""
        input_tensor = torch.randn((10, self.input_dim))  # Batch of 10 samples
        mixture_weights, component_params = self.model(input_tensor)

        # Ensure it produces outputs with correct shapes
        self.assertEqual(mixture_weights.shape, (10, self.output_components))
        self.assertTrue(0 <= mixture_weights.min() and mixture_weights.max() <= 1)

        # Ensure that mixture weights sum to 1 for each sample in the batch
        self.assertTrue(torch.allclose(mixture_weights.sum(dim=1), torch.ones(10), atol=1e-5))

    def test_weight_distributions(self):
        """Test that weights are sampled from distributions."""
        for mean, log_std in self.model.layers:
            self.assertFalse(torch.allclose(mean, torch.zeros_like(mean)))
            self.assertTrue(torch.all(log_std <= 0))  # Since we initialized log_std to zeros, it should be non-positive

    def test_dmm_output(self):
        """Test the output specifics of the Dirichlet Mixture Model."""
        input_tensor = torch.randn((10, self.input_dim))
        mixture_weights, component_params = self.model(input_tensor)

        # Ensure mixture weights are correctly normalized
        self.assertTrue(torch.allclose(mixture_weights.sum(dim=1), torch.ones(10), atol=1e-5))

        # For this test, we aren't specifying the nature of component_params.
        # But if we assume Gaussian components, we could add tests to ensure mean and variance values make sense.

    # Additional tests can be added as the model gets more functionality (e.g., training, prediction, inference)

    def test_train_method(self):
        # Define dataset and data loader parameters
        NUM_SAMPLES = 1000
        INPUT_DIM = 5  # Adjust as per your needs
        OUTPUT_DIM = 3  # Adjust as per your needs
        BATCH_SIZE = 32
        # Create dataset and data loader
        dummy_dataset = DummyDataset(NUM_SAMPLES, INPUT_DIM, OUTPUT_DIM)
        data_loader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE, shuffle=True)
        initial_loss = self.model.train(data_loader, epochs=1)
        final_loss = self.model.train(data_loader, epochs=5)
        self.assertTrue(final_loss < initial_loss)

class DummyDataset(Dataset):
    def __init__(self, num_samples, input_dim, output_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Generate random data for inputs and outputs
        self.inputs = torch.randn(num_samples, input_dim)
        self.outputs = torch.randn(num_samples, output_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]



if __name__ == '__main__':
    unittest.main()

