Gausimport torch.nn as nn
import itertools
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt

class SyntheticDataGenerator:
    def __init__(self, S, N, K, gaussian_width_factor=0.1):
        self.S = S
        self.N = N
        self.K = K
        self.gaussian_width_factor = gaussian_width_factor
        self.transition_function = self.simple_transition

    def simple_transition(self, means):
        """Simple transition function based on modulo operation"""
        x = np.random.randint(1, 2)
        if x == 1:
            return [mean + 1 for mean in means]
        if x == 2:
            return [mean + 2 for mean in means]
        if x == 3:
            return [mean + 3 for mean in means]

    def complex_transition(self, means):
        """Complex transition function as described earlier"""
        new_means = []
        for i in range(self.N):
            new_mean = int(means[i] * np.sin(means[(i + 1) % self.N]) +
                          means[(i + 2) % self.N] * np.cos(means[(i + 3) % self.N]))
            new_means.append(new_mean % self.K)
        return new_means

    def set_transition_function(self, func_name):
        if func_name == "simple":
            self.transition_function = self.simple_transition
        else:
            self.transition_function = self.complex_transition

    def generate_gaussian(self, mean, std, length):
        """Generate a Gaussian-like distribution tailored for a discrete setting."""
        x = np.arange(length)
        gaussian = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return gaussian / gaussian.sum()

    def generate(self):
        input_data = []
        output_data = []

        for _ in range(self.S):
            input_sample = []
            means = []

            # Generate the input sample
            for _ in range(self.N):
                mean = np.random.randint(0, self.K)
                std = self.K * self.gaussian_width_factor  # control the width of the Gaussian
                dist = self.generate_gaussian(mean, std, self.K)
                input_sample.append(dist)
                means.append(mean)

            new_means = self.transition_function(means)
            output_sample = []
            for i in range(self.N):
                std = self.K * self.gaussian_width_factor
                dist = self.generate_gaussian(new_means[i], std, self.K)
                output_sample.append(dist)

            input_data.append(input_sample)
            output_data.append(output_sample)

        return torch.from_numpy(np.array(input_data)).float(), torch.from_numpy(np.array(output_data)).float()

def plot_heatmaps(targets, predictions, N, K, num_samples=5):
    """
    Plots target, predicted, and difference heatmaps,
    along with bar charts for each of the N dimensions for both target and prediction.

    Parameters:
    - targets: A tensor of target values.
    - predictions: A tensor of predicted values.
    - N: Number of dimensions.
    - K: Number of classes.
    - num_samples: Number of random samples to plot.
    """
    # Ensure we don't exceed the available samples
    num_samples = min(num_samples, len(predictions))

    # Randomly select sample indices
    indices = torch.randint(0, len(predictions), (num_samples,))

    # Set up the figure
    fig, axs = plt.subplots(num_samples * (3), 1, figsize=(5, 4 * num_samples * (3)))

    for i, idx in enumerate(indices):
        # Extract the data for the current index
        target_sample = targets[idx].detach().numpy()
        predicted_sample = predictions[idx].detach().numpy()
        difference = target_sample - predicted_sample

        # Compute row sums for annotations
        target_row_sums = target_sample.sum(axis=1)
        predicted_row_sums = predicted_sample.sum(axis=1)

        # Plot the target heatmap with row sums
        axs[i * (3 + N * 2)].imshow(target_sample, cmap='viridis', aspect='auto')
        axs[i * (3 + N * 2)].set_title(f"Target Sample {i + 1}")
        axs[i * (3 + N * 2)].axis('off')
        for j, row_sum in enumerate(target_row_sums):
            axs[i * (3 + N * 2)].text(K + 0.5, j, f"Sum: {row_sum:.2f}", va='center')

        # Plot the predicted heatmap with row sums
        axs[i * (3 + N * 2) + 1].imshow(predicted_sample, cmap='viridis', aspect='auto')
        axs[i * (3 + N * 2) + 1].set_title(f"Predicted Sample {i + 1}")
        axs[i * (3 + N * 2) + 1].axis('off')
        for j, row_sum in enumerate(predicted_row_sums):
            axs[i * (3 + N * 2) + 1].text(K + 0.5, j, f"Sum: {row_sum:.2f}", va='center')

        # Plot the difference heatmap
        axs[i * (3 + N * 2) + 2].imshow(difference, cmap='viridis', aspect='auto')
        axs[i * (3 + N * 2) + 2].set_title(f"Difference Sample {i + 1}")
        axs[i * (3 + N * 2) + 2].axis('off')

    plt.tight_layout()
    plt.show()


# 1. Gaussian Filter Creation

def compute_gaussian_means(K, L):
    return [(K / (L + 1)) * i for i in range(1, L + 1)]


def gaussian_filter(K, mu, sigma):
    x = np.arange(1, K + 1)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    return gaussian / gaussian.sum()


def generate_filters_for_dimension(K, L, sigma):
    means = compute_gaussian_means(K, L)
    filters = [gaussian_filter(K, mu, sigma) for mu in means]
    return filters


def generate_filter_combinations(K, L, sigma, N):
    # Generate 1D Gaussian filters for the K dimension
    filters_for_K = generate_filters_for_dimension(K, L, sigma)

    # Generate all possible combinations of these 1D filters across the N dimensions
    filter_combinations = list(itertools.product(filters_for_K, repeat=N))

    # Convert the combinations into 2D filters of shape N x K
    combined_filters = [np.vstack(comb) for comb in filter_combinations]

    return combined_filters

class CustomConvolutionLayer(nn.Module):
    def __init__(self, filters):
        super(CustomConvolutionLayer, self).__init__()
        self.filters = [torch.tensor(f, dtype=torch.float32) for f in filters]

    def forward(self, x):
        outputs = []
        for f in self.filters:
            output = torch.sum(x * f)
            output = output / 5
            outputs.append(output)
        return torch.stack(outputs)

# 2. Neural Network Layers
class LatentDistributionLayer(nn.Module):
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

class ModifiedCustomCNN(nn.Module):
    def __init__(self, filters, latent_dim, output_shape):
        super(ModifiedCustomCNN, self).__init__()
        self.conv = CustomConvolutionLayer(filters)
        self.latent = LatentDistributionLayer(len(filters), latent_dim)
        self.fc = FullyConnectedLayer(latent_dim, output_shape)
        self.rowwise_softmax = RowwiseSoftmaxLayer()

    def forward(self, x):
        x = self.conv(x)
        mu, logvar = self.latent(x)
        z = self.latent.reparameterize(mu, logvar)
        x = self.fc(z)
        return self.rowwise_softmax(x), mu, logvar


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_shape):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_shape[0] * output_shape[1])
        self.output_shape = output_shape

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, *self.output_shape)


class RowwiseSoftmaxLayer(nn.Module):
    def __init__(self):
        super(RowwiseSoftmaxLayer, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(x)


class KLDivergenceRowwiseLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceRowwiseLoss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, predicted, target):
        predicted_log_softmax = torch.log_softmax(predicted, dim=2)
        total_loss = 0
        for i in range(predicted.shape[2]):
            total_loss += self.kl_div(predicted_log_softmax[:, :, i], target[:, :, i])
        return total_loss

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class RegularizedKLDIV(nn.Module):
    def __init__(self, model, lambda_reg=0.00001, lambda_latent=0.00005):
        super(RegularizedKLDIV, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.model = model
        self.lambda_reg = lambda_reg
        self.lambda_latent = lambda_latent

    def forward(self, predicted, target, mu, logvar):
        predicted_log_softmax = torch.log_softmax(predicted, dim=2)
        total_loss = 0
        for i in range(predicted.shape[2]):
            total_loss += self.kl_div(predicted_log_softmax[:, :, i], target[:, :, i])

        # L1 regularization for the linear layer weights
        l1_reg = 0
        for param in self.model.latent.mu_layer.parameters():
            l1_reg += torch.norm(param, 1)

        total_loss += self.lambda_reg * l1_reg

        # KL divergence from latent layer
        kl_latent = kl_divergence(mu, logvar)

        total_loss += self.lambda_latent * kl_latent

        return total_loss

class Loss(nn.Module):
    def __init__(self, model, lambda_latent=0.05):
        super(RegularizedKLDIV, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.model = model
        self.lambda_latent = lambda_latent

    def forward(self, predicted, target, mu, logvar):
        predicted_log_softmax = torch.log_softmax(predicted, dim=1)
        total_loss = 0
        for i in range(predicted.shape[1]):
            total_loss += self.kl_div(predicted_log_softmax[:, i, :], target[:, i, :])

        # KL divergence from latent layer
        kl_latent = kl_divergence(mu, logvar)

        total_loss += self.lambda_latent * kl_latent

        return total_loss


# 3. Full Model

class CustomCNN(nn.Module):
    def __init__(self, filters, latent_dim, output_shape):
        super(CustomCNN, self).__init__()
        self.conv = CustomConvolutionLayer(filters)
        self.latent = LatentDistributionLayer(len(filters), latent_dim)
        self.fc = FullyConnectedLayer(latent_dim, output_shape)
        self.rowwise_softmax = RowwiseSoftmaxLayer()

    def forward(self, x):
        x = self.conv(x)
        mu, logvar = self.latent(x)
        z = self.latent.reparameterize(mu, logvar)
        x = self.fc(z)
        return self.rowwise_softmax(x), mu, logvar


# 4. Training Loop

def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs)

            # Updated loss function to include mu and logvar
            loss = loss_fn(outputs, targets, mu, logvar)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, mu, logvar = model(inputs)

                # Updated loss function to include mu and logvar
                loss = loss_fn(outputs, targets, mu, logvar)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    plot_heatmaps(targets, outputs, 8, 8, num_samples=5)

    return train_losses, val_losses


# 5. Main Execution

if __name__ == "__main__":
    N = 8
    K = 8
    L = 4
    std = 0.05
    num_train_samples = 1000
    num_val_samples = 200
    latent_dim = 8
    epochs = 100
    lr = 0.001

    Syn = SyntheticDataGenerator(num_train_samples, N, K, std)
    X_train, Y_train = Syn.generate()
    Syn2 = SyntheticDataGenerator(num_val_samples, N, K, std)
    X_val, Y_val = Syn2.generate()
    print("Data Generated")

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model, optimizer, and loss function
    filter_combinations = generate_filter_combinations(N, L, std, K)
    print("Filters Generated")
    model = ModifiedCustomCNN(filter_combinations, latent_dim, (N, K))

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    #loss_fn = KLDivergenceRowwiseLoss()
    loss_fn = RegularizedKLDIV(model)

    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=epochs)

    # Plot losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
