import torch
import torch.nn as nn
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist


class StochasticProcessNeuralModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, latent_dim):
        super(StochasticProcessNeuralModel, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # 2 for mean and variance
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Parameters for the guide
        self.w_loc = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.w_scale = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.b_loc = nn.Parameter(torch.randn(hidden_dim))
        self.b_scale = nn.Parameter(torch.randn(hidden_dim))

    def model(self, x, y=None):
        # Pyro sample statements for the prior
        w_prior = dist.Normal(loc=torch.zeros_like(self.w_loc), scale=torch.ones_like(self.w_scale)).to_event(2)
        b_prior = dist.Normal(loc=torch.zeros_like(self.b_loc), scale=torch.ones_like(self.b_scale)).to_event(1)

        priors = {'w': w_prior, 'b': b_prior}
        lifted_module = pyro.random_module("module", self.decoder, priors)
        lifted_reg_model = lifted_module()

        with pyro.plate("data", x.shape[0]):
            # Run the encoder
            z_loc, z_scale = self.encoder(x).chunk(2, dim=-1)
            z_scale = nn.functional.softplus(z_scale)  # Ensure positive values
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            # Decoder
            prediction_mean = lifted_reg_model(z)
            pyro.sample("obs", dist.Normal(prediction_mean, 0.1).to_event(1), obs=y)

    def guide(self, x, y=None):
        # Pyro sample statements for the guide
        w_loc = self.w_loc
        w_scale = nn.functional.softplus(self.w_scale)  # Ensure positive values
        b_loc = self.b_loc
        b_scale = nn.functional.softplus(self.b_scale)  # Ensure positive values

        w_dist = dist.Normal(loc=w_loc, scale=w_scale).to_event(2)
        b_dist = dist.Normal(loc=b_loc, scale=b_scale).to_event(1)

        mw_param = pyro.sample("w", w_dist)
        mb_param = pyro.sample("b", b_dist)

        return mw_param, mb_param


"""
# Initialize the model
model_instance_v2 = StochasticProcessNeuralModel_v2(input_dim=10, hidden_dim=20, output_dim=10, latent_dim=5)

# SVI (Stochastic Variational Inference) for training
optimizer = Adam({"lr": 0.001})
svi_v2 = SVI(model_instance_v2.model, model_instance_v2.guide, optimizer, loss=Trace_ELBO())

# Number of samples
num_samples = 1000

# Input data: Random 10-dimensional vectors
input_data = torch.randn(num_samples, 10)

# Convert input data into valid probability distributions using softmax
input_data = torch.nn.functional.softmax(input_data, dim=-1)

# Output data: Simple transformation of input data with added noise
transformation_matrix = torch.randn(10, 10)
output_data = input_data @ transformation_matrix + 0.1 * torch.randn(num_samples, 10)

# Splitting the data into training and validation sets (80-20 split)
train_size = int(0.8 * num_samples)
train_input, val_input = input_data[:train_size], input_data[train_size:]
train_output, val_output = output_data[:train_size], output_data[train_size:]

# Training loop
num_epochs = 1000
train_losses_v2 = []
for epoch in range(num_epochs):
    # Forward + Backward pass + Optimize
    loss = svi_v2.step(train_input, train_output)
    train_losses_v2.append(loss)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
"""

