import torch
import pyro
import pyro.distributions as dist
from torch import nn
from pyro import sample, plate


# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_var=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var

        # Weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Initialize weights and biases
        self.weight.data.normal_(0, prior_var**0.5)
        self.bias.data.normal_(0, prior_var**0.5)

    def forward(self, x):
        return torch.mm(x, self.weight.t()) + self.bias


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, prior_var=1.0):
        super(Encoder, self).__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim, prior_var)
        self.fc2 = BayesianLinear(hidden_dim, latent_dim, prior_var)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, prior_var=1.0):
        super(Decoder, self).__init__()
        self.fc1 = BayesianLinear(latent_dim, hidden_dim, prior_var)
        self.fc2 = BayesianLinear(hidden_dim, output_dim, prior_var)
        self.relu = nn.ReLU()

    def forward(self, r):
        r = self.relu(self.fc1(r))
        output = self.fc2(r)
        return torch.nn.functional.softmax(output, dim=-1)  # Applying softmax over the last dimension


# Main Model
class StochasticProcessNeuralModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, latent_dim, prior_var=1.0):
        super(StochasticProcessNeuralModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, prior_var)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim, prior_var)
        self.prior_var = prior_var

    def model(self, x, y=None):
        w_prior = dist.Normal(loc=torch.zeros_like(self.encoder.fc1.weight), scale=torch.ones_like(self.encoder.fc1.weight))
        b_prior = dist.Normal(loc=torch.zeros_like(self.encoder.fc1.bias), scale=torch.ones_like(self.encoder.fc1.bias))
        priors = {'fc1.weight': w_prior, 'fc1.bias': b_prior}
        lifted_module = pyro.random_module("module", self.encoder, priors)
        lifted_reg_model = lifted_module()

        with plate("data", x.size(0)):
            r = lifted_reg_model(x)
            prediction_mean = self.decoder(r)
            pyro.sample("obs", dist.Normal(prediction_mean, 0.1 * torch.ones(x.size(0))), obs=y)

    def forward(self, x):
        r = self.encoder(x)
        return self.decoder(r)

    def guide(self, x, y=None):
        w_loc = torch.randn_like(self.encoder.fc1.weight)
        w_scale = torch.randn_like(self.encoder.fc1.weight)
        b_loc = torch.randn_like(self.encoder.fc1.bias)
        b_scale = torch.randn_like(self.encoder.fc1.bias)
        w_loc_param = pyro.param("w_loc", w_loc)
        w_scale_param = pyro.param("w_scale", w_scale, constraint=dist.constraints.positive)
        b_loc_param = pyro.param("b_loc", b_loc)
        b_scale_param = pyro.param("b_scale", b_scale, constraint=dist.constraints.positive)
        w_prior = dist.Normal(loc=w_loc_param, scale=w_scale_param)
        b_prior = dist.Normal(loc=b_loc_param, scale=b_scale_param)
        priors = {'fc1.weight': w_prior, 'fc1.bias': b_prior}
        lifted_module = pyro.random_module("module", self.encoder, priors)
        return lifted_module()


# Sample Test
def test_forward_pass():
    x = torch.randn(10, 20)  # 10 samples, 20 dimensions
    model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)
    output = model(x)
    assert output is not None
    assert output.shape == (10, 20)

# Run the test
test_forward_pass()


def test_initialization():
    try:
        model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)
        assert isinstance(model, StochasticProcessNeuralModel)
    except Exception as e:
        raise AssertionError(f"Initialization failed with error: {e}")


def test_forward_pass():
    x = torch.randn(10, 20)  # 10 samples, 20 dimensions
    model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)
    output = model(x)

    assert output is not None, "Output is None"
    assert output.shape == (10, 20), f"Expected output shape (10, 20), but got {output.shape}"


def test_backward_pass():
    x = torch.randn(10, 20)
    y = torch.randn(10, 20)
    model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # If no errors, pass the test


def test_variable_input_size():
    sizes = [(5, 20), (15, 20), (20, 20)]
    model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)

    for size in sizes:
        x = torch.randn(*size)
        output = model(x)
        assert output.shape == size, f"For input shape {size}, expected output shape {size}, but got {output.shape}"


def test_distribution_handling():
    x = torch.randn(10, 20)
    x = torch.nn.functional.softmax(x, dim=1)
    model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)
    output = model(x)

    # Ensure output values are between 0 and 1 and sum to 1
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output values are not between 0 and 1"
    assert torch.allclose(output.sum(dim=1), torch.tensor([1.0] * 10), atol=1e-4), "Output values do not sum to 1"


def test_forward_inference():
    x = torch.randn(10, 20)
    model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)
    output = model(x)

    # Ensure output is a valid probability distribution
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output values are not between 0 and 1"
    assert torch.allclose(output.sum(dim=1), torch.tensor([1.0] * 10), atol=1e-4), "Output values do not sum to 1"


def test_training_time():
    import time

    x = torch.randn(100, 20)
    y = torch.randn(100, 20)
    model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    start_time = time.time()
    for _ in range(100):  # 100 training iterations
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()

    elapsed_time = end_time - start_time
    assert elapsed_time < 60, f"Training took too long: {elapsed_time} seconds"  # example threshold


def test_inference_time():
    import time

    x = torch.randn(100, 20)
    model = StochasticProcessNeuralModel(input_dim=20, hidden_dim=30, output_dim=20, latent_dim=15)

    start_time = time.time()
    for _ in range(100):  # 100 inference iterations
        output = model(x)
    end_time = time.time()

    elapsed_time = end_time - start_time
    assert elapsed_time < 30, f"Inference took too long: {elapsed_time} seconds"  # example threshold


# Run the tests
test_initialization()
test_forward_pass()
test_backward_pass()
test_variable_input_size()
test_distribution_handling()
test_forward_inference()
test_training_time()
test_inference_time()


import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO


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

# Initialize the model
model_instance = StochasticProcessNeuralModel(input_dim=10, hidden_dim=20, output_dim=10, latent_dim=5)

# Optimizer
optimizer = optim.Adam({"lr": 0.01})

# SVI (Stochastic Variational Inference) for training
svi = SVI(model_instance.model, model_instance.guide, optimizer, loss=Trace_ELBO())

# Training loop
num_epochs = 500
train_losses = []

for epoch in range(num_epochs):
    # Forward + Backward pass + Optimize
    loss = svi.step(train_input, train_output)
    train_losses.append(loss)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

# Return the training losses for visualization
train_losses[-10:]  # Display the last 10 training losses
