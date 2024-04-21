# 1. Gaussian Filter Creation
import numpy as np
import itertools
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def compute_gaussian_means(K, L):
    return [(K / (L + 1)) * i for i in range(1, L + 1)]


def gaussian_filter(K, mu, sigma):
    x = np.arange(1, K + 1)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    return gaussian / gaussian.sum()


def cauchy_filter(K, mu, gamma):
    """
    Create a Cauchy filter of length K.
    mu: location parameter
    gamma: scale parameter
    """
    x = np.arange(1, K + 1)
    cauchy = 1 / (np.pi * gamma * (1 + ((x - mu) / gamma)**2))
    return cauchy / cauchy.sum()

def generate_filters_for_dimension(K, L, sigma, filter_type="cauchy"):
    means = compute_gaussian_means(K, L)
    if filter_type == "gaussian":
        filters = [gaussian_filter(K, mu, sigma) for mu in means]
    elif filter_type == "cauchy":
        filters = [cauchy_filter(K, mu, sigma) for mu in means]
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    return filters

def generate_filter_combinations(K, L, sigma, N, filter_type="cauchy"):
    # Generate 1D filters for the K dimension based on the filter type
    filters_for_K = generate_filters_for_dimension(K, L, sigma, filter_type)

    # Generate all possible combinations of these 1D filters across the N dimensions
    filter_combinations = list(itertools.product(filters_for_K, repeat=N))

    # Convert the combinations into 2D filters of shape N x K
    combined_filters = [np.vstack(comb) for comb in filter_combinations]

    return combined_filters


class SyntheticDataGenerator:
    def __init__(self, S, N, K, gamma=0.5, context_size=None):
        self.S = S
        self.N = N
        self.K = K
        self.gamma = gamma  # Using gamma instead of width_factor
        self.context_size = context_size
        self.transition_function = self.simple_transition

    def simple_transition(self, means):
        """Simple transition function based on addition."""
        return [mean + 1 for mean in means]

    def generate_distribution(self, mean, gamma, length):
        """Generate a simple Cauchy-like distribution tailored for a discrete setting."""
        x = np.linspace(0, length-1, length)
        cauchy = 1 / (np.pi * gamma * (1 + ((x - mean) / gamma)**2))
        return cauchy / cauchy.sum()

    def generate(self):
        input_data = []
        output_data = []

        for _ in range(self.S):
            if self.context_size:  # When context_size is provided
                batch_input = []
                batch_output = []
                for _ in range(self.context_size):
                    input_sample, output_sample = self._generate_sample()
                    batch_input.append(input_sample)
                    batch_output.append(output_sample)
                input_data.append(batch_input)
                output_data.append(batch_output)
            else:  # For backward compatibility, when context_size is None
                input_sample, output_sample = self._generate_sample()
                input_data.append(input_sample)
                output_data.append(output_sample)

        return torch.from_numpy(np.array(input_data)).float(), torch.from_numpy(np.array(output_data)).float()

    def _generate_sample(self):
        """Helper function to generate a single sample."""
        input_sample = []
        output_sample = []

        # Generate the means for the entire sample first
        means = [np.random.randint(0, self.K - 5) for _ in range(self.N)]
        # Calculate the sum of the means
        total_mean = sum(means)
        # Based on the collective means, determine the transitions
        for mean in means:
            x = np.random.randint(1, 3)
            if x == 1:
                new_mean = 5
                if total_mean == 14*0.5:
                    new_mean = 1 + mean
                if total_mean == 14*0.5:
                    new_mean = 2 + mean
                if total_mean == 16*0.5:
                    new_mean = 3 + mean
                if total_mean == 12*0.5:
                    new_mean = 4 - mean
                if total_mean == 18*0.5:
                    new_mean = 5
                if total_mean == 12*0.5:
                    new_mean = 6 - mean
                if total_mean == 10*0.5:
                    new_mean = 7 - mean
                if total_mean == 20*0.5:
                    new_mean = 8 - mean*0.5
                if total_mean == 18*0.5:
                    new_mean = 9 - mean*0.5
                if total_mean == 16*0.5:
                    new_mean = 10 + mean*0.15
                if total_mean > 20:
                    new_mean = 11 - mean
                if total_mean < 10*0.5:
                    new_mean = 12 + mean*0.15
                if new_mean > 11:
                    new_mean = 11
                if new_mean < 0:
                    new_mean = 0
            if x == 2:
                new_mean=7
                if total_mean == 16*0.5:
                    new_mean = 3 + mean
                if total_mean == 14*0.5:
                    new_mean = 4 + mean
                if total_mean == 18*0.5:
                    new_mean = 5 + mean
                if total_mean == 12*0.5:
                    new_mean = 6 - mean
                if total_mean == 14*0.5:
                    new_mean = 7
                if total_mean == 12*0.5:
                    new_mean = 8 - mean
                if total_mean == 10*0.5:
                    new_mean = 9 - mean
                if total_mean == 20*0.5:
                    new_mean = 10 - mean*0.5
                if total_mean == 18*0.5:
                    new_mean = 11 - mean*0.5
                if total_mean == 16*0.5:
                    new_mean = 12 + mean*0.15
                if total_mean > 20*0.5:
                    new_mean = 13 - mean
                if total_mean < 10:
                    new_mean = 14 + mean*0.15
                if new_mean > 11:
                    new_mean = 11
                if new_mean < 0:
                    new_mean = 2

                new_mean = np.around(new_mean)


            dist_i = self.generate_distribution(mean, self.gamma, self.K)
            dist_0 = self.generate_distribution(new_mean, self.gamma, self.K)

            input_sample.append(dist_i)
            output_sample.append(dist_0)

        return input_sample, output_sample


def plot_heatmaps(input_data, targets, predictions, N, K, L, num_samples=5, plot=True):
    """
    Plots input, target, predicted, and difference heatmaps,
    along with bar charts for each of the N dimensions for both target and prediction.

    Returns:
    - mean and std of the distribution of all_argmax_diffs.

    Parameters:
    - input_data: A tensor of input values.
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

    all_argmax_diffs = []  # Store all differences for all samples

    if plot == False:
        for idx in range(len(predictions)):
            # Extract the data for the current index
            target_sample = targets[idx].detach().numpy()
            predicted_sample = predictions[idx].detach().numpy()

            for dim in range(N):
                # Compute argmax differences for each dimension
                target_argmax = np.argmax(target_sample[dim], axis=0)  # Adjusted axis
                predicted_argmax = np.argmax(predicted_sample[dim], axis=0)  # Adjusted axis
                argmax_differences = np.abs(target_argmax - predicted_argmax)
            all_argmax_diffs.append(argmax_differences)

        mean_diff = np.mean(all_argmax_diffs)
        std_diff = np.std(all_argmax_diffs)

        return mean_diff, std_diff

    fig, axs = plt.subplots(num_samples * (4), 1, figsize=(5, 4 * num_samples * (4)))


    for i, idx in enumerate(indices):
        # Extract the data for the current index
        input_sample = input_data[idx].detach().numpy()
        target_sample = targets[idx].detach().numpy()
        predicted_sample = predictions[idx].detach().numpy()
        difference = target_sample - predicted_sample

        # Compute row sums for annotations
        input_row_sums = input_sample.sum(axis=1)
        target_row_sums = target_sample.sum(axis=1)
        predicted_row_sums = predicted_sample.sum(axis=1)

        # Plot the input heatmap with row sums
        axs[i * 4].imshow(input_sample, cmap='viridis', aspect='auto')
        axs[i * 4].set_title(f"Input Sample {i + 1}")
        axs[i * 4].axis('off')
        for j, row_sum in enumerate(input_row_sums):
            axs[i * 4].text(K + 0.5, j, f"Sum: {row_sum:.2f}", va='center')

        # Plot the target heatmap with row sums
        axs[i * 4 + 1].imshow(target_sample, cmap='viridis', aspect='auto')
        axs[i * 4 + 1].set_title(f"Target Sample {i + 1}")
        axs[i * 4 + 1].axis('off')
        for j, row_sum in enumerate(target_row_sums):
            axs[i * 4 + 1].text(K + 0.5, j, f"Sum: {row_sum:.2f}", va='center')

        # Plot the predicted heatmap with row sums
        axs[i * 4 + 2].imshow(predicted_sample, cmap='viridis', aspect='auto')
        axs[i * 4 + 2].set_title(f"Predicted Sample {i + 1}")
        axs[i * 4 + 2].axis('off')
        for j, row_sum in enumerate(predicted_row_sums):
            axs[i * 4 + 2].text(K + 0.5, j, f"Sum: {row_sum:.2f}", va='center')

        # Plot the difference heatmap
        axs[i * 4 + 3].imshow(difference, cmap='viridis', aspect='auto')
        axs[i * 4 + 3].set_title(f"Difference Sample {i + 1}")
        axs[i * 4 + 3].axis('off')

    for idx in range(len(predictions)):
        # Extract the data for the current index
        target_sample = targets[idx].detach().numpy()
        predicted_sample = predictions[idx].detach().numpy()

        for dim in range(N):
            # Compute argmax differences for each dimension
            target_argmax = np.argmax(target_sample[dim], axis=0)  # Adjusted axis
            predicted_argmax = np.argmax(predicted_sample[dim], axis=0)  # Adjusted axis
            argmax_differences = target_argmax - predicted_argmax
            all_argmax_diffs.append(argmax_differences)

    mean_diff = np.mean(all_argmax_diffs)
    std_diff = np.std(all_argmax_diffs)

    plt.tight_layout()
    plt.show()

    # Plot the histogram for the differences
    plt.hist(all_argmax_diffs, bins=30, edgecolor='black')
    plt.title("Distribution of Differences")
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.show()

    return mean_diff, std_diff
