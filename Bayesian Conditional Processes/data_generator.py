import torch

class SyntheticDataGenerator:
    """
    A class to generate synthetic data using a Gaussian Mixture Model.

    Attributes:
    - F: Dimension of input distribution.
    - K: Discretization for input distribution.
    - H: Dimension of Gaussian mixture.
    - G: Number of Gaussian mixtures.
    - J: Number of Gaussian mixtures a point in the input space should point towards.
    - means: Mean for each Gaussian mixture.
    - covariances: Covariance matrix for each Gaussian mixture.
    - transition_matrix: Transition probabilities matrix.
    """

    def __init__(self, F, K, H, G, J, transition_matrix=None):
        """
        Initialize the SyntheticDataGenerator with given parameters.

        Parameters:
        - F (int): Dimension of input distribution.
        - K (int): Discretization for input distribution.
        - H (int): Dimension of Gaussian mixture.
        - G (int): Number of Gaussian mixtures.
        - J (int): Number of Gaussian mixtures a point in the input space should point towards.
        - transition_matrix (torch.Tensor, optional): Pre-defined transition probabilities matrix.
          If None, it will be initialized.
        """
        self.F = F
        self.K = K
        self.H = H
        self.G = G
        self.J = J
        self.means = torch.rand(G, H)
        self.covariances = [torch.eye(H) for _ in range(G)]
        if transition_matrix is None:
            self.transition_matrix = self.initialize_transition_matrix()
        else:
            self.transition_matrix = transition_matrix

    def initialize_transition_matrix(self):
        """
        Initialize the transition matrix based on the J-modal and smoothness strategy.

        Returns:
        - transition_matrix (torch.Tensor): Initialized transition probabilities matrix of shape G x F x K.
        """
        transition_matrix = torch.zeros(self.G, self.F, self.K)
        for f in range(self.F):
            for k in range(self.K):
                distances = torch.norm(self.means[:, :2] - torch.tensor([f, k]), dim=1)
                _, closest_mixtures = torch.topk(-distances, self.J)  # Negate distances to get largest values with topk
                transition_matrix[closest_mixtures, f, k] = 1 / distances[closest_mixtures]
                transition_matrix[:, f, k] /= transition_matrix[:, f, k].sum()
        return transition_matrix

    def sample_input_distribution(self):
        """
        Sample from an input distribution of dimension F and discretization K.

        Returns:
        - torch.Tensor: Sampled values from the input distribution.
        """
        return torch.randint(0, self.K, (self.F,))

    def calculate_transition_probabilities(self, input_sample):
        """
        Modified method for calculate_transition_probabilities.
        Calculate conditional probabilities for transitioning to a specific Gaussian mixture
        using the conditional transition matrix and the input sample.

        Parameters:
        - input_sample (torch.Tensor): Sampled values from the input distribution.

        Returns:
        - torch.Tensor: Transition probabilities for the given input sample.
        """
        probs = self.transition_matrix[:, torch.arange(self.F), input_sample]
        normalized_probs = probs.mean(dim=1)  # Take the mean across dimensions
        if normalized_probs.sum() == 0:
            normalized_probs = torch.ones(self.G) / self.G  # Return uniform probabilities if the sum is zero
        normalized_probs = normalized_probs / normalized_probs.sum()

        return normalized_probs

    def sample_from_transition_probabilities(self, transition_probs):
        """
        Sample from the transition probabilities to determine which Gaussian mixture is chosen.

        Parameters:
        - transition_probs (torch.Tensor): Transition probabilities.

        Returns:
        - int: Index of the chosen Gaussian mixture.
        """
        return torch.multinomial(transition_probs, 1).item()

    def sample_from_mixture(self, mixture_idx):
        """
        Given a mixture index, sample from the Gaussian mixture and return its distribution.

        Parameters:
        - mixture_idx (int): Index of the chosen Gaussian mixture.

        Returns:
        - torch.Tensor: Sampled values from the chosen Gaussian mixture.
        """
        mean = self.means[mixture_idx]
        cov = self.covariances[mixture_idx]  # Use the 2D covariance matrix directly
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.sample((self.K,)).t()

    def generate_synthetic_data(self):
        # Sample from the input distribution
        input_sample_raw = self.sample_input_distribution().float()

        # Apply softmax to convert to probability
        input_sample = torch.nn.functional.softmax(input_sample_raw, dim=0)
        normalized_input_sample = input_sample.unsqueeze(-1).repeat(1, self.K)

        transition_probs = self.calculate_transition_probabilities(torch.round(input_sample * (self.K - 1)).long())
        chosen_mixture = self.sample_from_transition_probabilities(transition_probs)

        samples = torch.stack([self.sample_from_mixture(chosen_mixture) for _ in range(100)])
        distribution = torch.mean(samples, dim=0)
        distribution /= distribution.sum(dim=1, keepdim=True)

        return (normalized_input_sample, distribution)
