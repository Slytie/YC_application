import torch
import torch.nn as nn
from gaussian import generate_filter_combinations, SyntheticDataGenerator, plot_heatmaps
import numpy as np
import matplotlib.pyplot as plt
from hyperparameter_optimization import optimize_hyperparameters


class CustomConvolutionLayer(nn.Module):
    def __init__(self, filters, N):
        super(CustomConvolutionLayer, self).__init__()
        # Convert filters to tensors with requires_grad set to False
        self.filters = [torch.tensor(f, dtype=torch.float32, requires_grad=False) for f in filters]
        self.N = N

    def forward(self, x):
        outputs = []
        for f in self.filters:
            output = torch.sum(x * f)
            output = output / N
            outputs.append(output)
        return torch.stack(outputs)

# 2. Neural Network Layers
class LatentDistributionLayer(nn.Module):
    """
    Transforms an input vector into a latent space representation characterized by mean (mu) and log variance (logvar).

    The LatentDistributionLayer is designed to encapsulate the variational autoencoder (VAE) style of latent
    representation, wherein an input is transformed into parameters of a Gaussian distribution in the latent space.
    This layer facilitates the "encoding" part of the VAE architecture, where observed data is mapped to a
    distribution in a lower-dimensional latent space. The reparameterization trick is employed to allow for
    differentiable sampling from this distribution.

    Underlying Mathematics:
        - Given an input vector v, the transformation into the latent space is achieved using two separate neural
          network layers: one for the mean (mu) and one for the log variance (logvar):
            mu = W_mu v + b_mu
            logvar = W_var v + b_var
          where W_mu, W_var are the weight matrices and b_mu, b_var are the bias vectors for the mean and log-variance
          layers, respectively.

        - The reparameterization trick is used for sampling from the latent distribution:
            z = mu + exp(0.5 * logvar) * epsilon
          where epsilon is noise drawn from a standard normal distribution.

    Attributes:
        mu_layer (nn.Module): Linear layer to compute the mean of the latent distribution.
        logvar_layer (nn.Module): Linear layer to compute the log variance of the latent distribution.

    Args:
        input_dim (int): Dimension of the input vector.
        latent_dim (int): Desired dimension of the latent space representation.

    Returns:
        tuple: Two torch.Tensors representing the mean and log variance of the latent distribution.

    Examples:
    """
    def __init__(self, input_dim, latent_dim):
        super(LatentDistributionLayer, self).__init__()
        self.mu_layer = nn.Linear(input_dim, latent_dim)
        self.logvar_layer = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_shape, hidden_size=None):
        super(FullyConnectedLayer, self).__init__()

        # If hidden_size is not provided, default to input_size
        if hidden_size is None:
            hidden_size = input_size

        # First linear layer
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Activation function (you can choose any other activation like nn.ReLU() if you prefer)
        self.activation = nn.ReLU()

        # Second linear layer
        self.fc = nn.Linear(hidden_size, output_shape[0] * output_shape[1])
        self.output_shape = output_shape

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc(x)
        return x.view(-1, *self.output_shape)

class RowwiseSoftmaxLayer(nn.Module):
    def __init__(self):
        super(RowwiseSoftmaxLayer, self).__init__()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        return self.softmax(x)


class Loss(nn.Module):
    def __init__(self, model, lambda_latent=0.5, kl=False, mse=True, cross=True):
        super(Loss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='mean')
        self.model = model
        self.lambda_latent = lambda_latent
        self.alpha = 0.02

    def kl_divergence(self, mu, logvar):
        x = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return x

    def forward(self, predicted, target, mu_xy, logvar_xy, mu_x, logvar_x, epoch, kl=True, mse=False, cross=False):

        total_loss = 0
        if kl == True:
            predicted = predicted.squeeze(dim=1)
            predicted = (predicted + self.alpha) / (self.alpha * K + 1)
            predicted_log = torch.log(predicted + 1e-10)  # small constant added for numerical stability

            for k in range(predicted.shape[0]):
                for i in range(predicted.shape[1]):
                    kl_loss = self.kl_div(predicted_log[k, i, :], target[k, i, :])
                    total_loss += kl_loss

        if mse == True:
            predicted = predicted.squeeze(dim=1)
            loss = self.mse(predicted, target)
            total_loss += loss


        # KL divergence from latent layer for both xy and x spaces
        kl_latent_xy = self.kl_divergence(mu_xy, logvar_xy)
        kl_latent_x = self.kl_divergence(mu_x, logvar_x)
        total_latent_kl = kl_latent_xy + kl_latent_x

        if epoch > 100:
            total_loss += self.lambda_latent * total_latent_kl

        return total_loss

class LatentCNP(nn.Module):
    def __init__(self, filters, latent_dim, output_shape, context_size, N, verbose=False):
        super(LatentCNP, self).__init__()
        self.context_size = context_size
        self.conv = CustomConvolutionLayer(filters, N)
        self.latent_xy = LatentDistributionLayer(2 * len(filters), latent_dim)  # For (x, y) pairs
        self.latent_x = LatentDistributionLayer(len(filters), latent_dim)  # For x target
        self.fc = FullyConnectedLayer(2 * latent_dim, output_shape, 20)  # Factor of 2 because of concatenation
        self.fc2 = FullyConnectedLayer(len(filters)*len(filters), (1, 2 * latent_dim), 20)
        self.fc1 = FullyConnectedLayer(len(filters), (1, len(filters)), 20)  # Factor of 2 because of concatenation
        self.rowwise_softmax = RowwiseSoftmaxLayer()
        self.verbose = verbose

    def forward(self, x_context, y_context, x_target):
        outputs = []  # List to store outputs for each batch element
        mus_xy = []
        logvars_xy = []
        mus_x = []
        logvars_x = []

        # Loop through each batch element
        for i in range(x_context.size(0)):
            x_c = x_context[i]
            y_c = y_context[i]
            x_t = x_target[i]

            z_xy_list = []
            #z_total = []
            for x, y in zip(x_c.split(1, dim=0), y_c.split(1, dim=0)):
                x = x.squeeze(dim=0)
                y = y.squeeze(dim=0)
                outx = self.conv(x)
                outy = self.conv(y)
                outx = self.fc1(outx)
                outy = self.fc1(outy)
                outy = outy.squeeze(dim=0)
                outx = outx.squeeze(dim=0)
                outy = outy.squeeze(dim=0)
                outx = outx.squeeze(dim=0)
                # Reshaping the tensors A and B for broadcasting and then adding them
                #output = outx.view(-1, 1) + outy.view(1, -1)

                out = torch.cat([outx, outy], dim=0)
                mu, logvar = self.latent_xy(out)
                z_xy = self.latent_xy.reparameterize(mu, logvar)
                z_xy_list.append(z_xy)
                #z_total.append(output)
                if self.verbose:
                    print(f"Shape after conv for x: {outx.shape}")
                    print(f"Shape after conv for y: {outy.shape}")
                    print(f"Shape after concatenation: {out.shape}")
                    print(f"Shape after latent_xy for mu: {mu.shape}")
                    print(f"Shape after latent_xy for logvar: {logvar.shape}")
                    print(f"Shape after reparameterization: {z_xy.shape}")

            z_avg = torch.mean(torch.stack(z_xy_list), dim=0)
            #z_avg= torch.mean(torch.stack(z_total), dim=0)

            out_target = self.conv(x_t)
            # out_target = out_target.view(-1, 1)
            mu_target, logvar_target = self.latent_x(out_target)
            z_x = self.latent_x.reparameterize(mu_target, logvar_target)
            if self.verbose:
                print(f"Shape after conv for target x: {out_target.shape}")
                print(f"Shape after latent_x for mu_target: {mu_target.shape}")
                print(f"Shape after latent_x for logvar_target: {logvar_target.shape}")
                print(f"Shape after reparameterization for target: {z_x.shape}")

            z_combined = torch.cat([z_avg, z_x], dim=0)
            #z_combined = torch.cat([z_avg, out_target], dim=0)
            x_out = self.fc(z_combined)
            if self.verbose:
                print(f"Shape after concatenation of z_avg and z_x: {z_combined.shape}")
                print(f"Shape after fully connected layer: {x_out.shape}")

            outputs.append(self.rowwise_softmax(x_out))
            mus_xy.append(mu)
            logvars_xy.append(logvar)
            mus_x.append(mu_target)
            logvars_x.append(logvar_target)

        output_batch = torch.stack(outputs)
        mu_xy_batch = torch.stack(mus_xy)
        logvar_xy_batch = torch.stack(logvars_xy)
        mu_x_batch = torch.stack(mus_x)
        logvar_x_batch = torch.stack(logvars_x)

        return output_batch, mu_xy_batch, logvar_xy_batch, mu_x_batch, logvar_x_batch


    # This is a pseudo code and may require adjustments based on the complete context and other parts of the actual code.
def train_model(model, loss_function, optimizer, epochs = 10, batch_size = 64, N = 4, K = 8, L=6, context_size = 16, data_sigma=0.1):
    model.train()

    for epoch in range(epochs):

        # Generate random data
        Syn = SyntheticDataGenerator(batch_size, N, K, data_sigma)
        x_target, y_target = Syn.generate()
        Syn2 = SyntheticDataGenerator(batch_size, N, K, data_sigma, context_size)
        x_context, y_context = Syn2.generate()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output, mu_xy, logvar_xy, mu_x, logvar_x = model(x_context, y_context, x_target)

        # Compute loss
        loss = loss_function(output, y_target, mu_xy, logvar_xy, mu_x, logvar_x, epoch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print loss for debugging
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        output = output.squeeze(dim=1)
        mean, std = plot_heatmaps(x_target, y_target, output, N, K, L, num_samples=1, plot=False)
        print(f"Standard deviants {std}, Mean {mean}")
        if epoch== epochs-1:
            output = output.squeeze(dim=1)
            mean, std = plot_heatmaps(x_target, y_target, output, N, K, L, num_samples=1, plot=True)
            print(f"Standard deviants {std}, Mean {mean}")


    return std, mean

h={
"N" : 4,
"K" : 12,
"L" : 8,
"context_size" : 32,
"lr" : 0.001,
"data_sigma" : 1,
"filter_sigma" : 1,
"batch_size" : 128,
"lambda_latent" : 0.001,
"epochs" : 222,
"mean" : 0.04,
"std" : 0.21
}

# Adaptive filtering though information gain. ie attention. Look at Free energy principle for information gain seeking.
# What about losses like the moments of the distribution, and cumulative distribution. Detection theory.
# Adding an initial filter to determine which weights to update.

N=h["N"]
K=h["K"]
L=h["L"]
data_sigma=h["data_sigma"]
filter_sigma = h["filter_sigma"]
batch_size=h["batch_size"]
lambda_latent=h["lambda_latent"]
context_size=h["context_size"]
lr=h["lr"]
epochs = h["epochs"]

filters = generate_filter_combinations(K, L, filter_sigma, N)

# Re-initialize the simplified CNP Model
model = LatentCNP(filters, latent_dim = 8, output_shape = (N, K), context_size=context_size, N=N)

# Re-initialize the optimizer for the simplified CNP model
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_function = Loss(model,lambda_latent=lambda_latent)

# Test the training function
std, mean = train_model(model, loss_function, optimizer, epochs=epochs, context_size= context_size, N=N, K=K, L=L,
                        data_sigma = data_sigma, batch_size = batch_size)


