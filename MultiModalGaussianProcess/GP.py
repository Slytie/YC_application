import torch
class GaussianProcessSpace:
    """
    A class to represent the space of potential inputs for a Gaussian Process given an output.

    Attributes:
    -----------
    X_train : torch.Tensor
        Training data.
    kernel_matrix : torch.Tensor
        Kernel matrix constructed using the training data.
    significant_eigenvectors : torch.Tensor
        Eigenvectors corresponding to significant eigenvalues of the kernel matrix.

    Methods:
    --------
    rbf_kernel(x, y):
        Computes the RBF kernel between two sets of points.
    compute_kernel_matrix():
        Computes the kernel matrix for the training data.
    perform_eigendecomposition():
        Performs eigendecomposition on the kernel matrix.
    identify_significant_eigenvectors(n):
        Identifies `n` eigenvectors corresponding to the top `n` eigenvalues.
    project_to_subspace(y):
        Projects an observed output onto the subspace spanned by significant eigenvectors.
    project_to_input_space():
        Projects back to the discrete input space to find representative points.
    """

    def __init__(self, X_train):
        """
        Initializes the GaussianProcessSpace with training data.

        Parameters:
        -----------
        X_train : torch.Tensor
            Training data.
        """
        self.X_train = X_train
        self.kernel_matrix = self.compute_kernel_matrix()
        self.significant_eigenvectors = None

    @staticmethod
    def rbf_kernel(x, y, sigma=1.0, lengthscale=1.0):
        """
        Computes the RBF kernel between two sets of points.

        Parameters:
        -----------
        x, y : torch.Tensor
            Input tensors.
        sigma : float
            Amplitude parameter.
        lengthscale : float
            Length scale parameter.

        Returns:
        --------
        torch.Tensor
            RBF kernel matrix.
        """
        sqdist = torch.cdist(x, y, p=2) ** 2
        return sigma ** 2 * torch.exp(-0.5 * sqdist / lengthscale ** 2)

    def compute_kernel_matrix(self):
        """
        Computes the kernel matrix for the training data.

        Returns:
        --------
        torch.Tensor
            Kernel matrix.
        """
        return self.rbf_kernel(self.X_train, self.X_train)

    def perform_eigendecomposition(self):
        """
        Performs eigendecomposition on the kernel matrix.

        Returns:
        --------
        eigenvalues, eigenvectors : torch.Tensor, torch.Tensor
            Eigenvalues and eigenvectors of the kernel matrix.
        """
        return torch.linalg.eigh(self.kernel_matrix)

    def identify_significant_eigenvectors(self, n):
        """
        Identifies `n` eigenvectors corresponding to the top `n` eigenvalues.

        Parameters:
        -----------
        n : int
            Number of significant eigenvectors to consider.
        """
        eigenvalues, eigenvectors = self.perform_eigendecomposition()
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        self.significant_eigenvectors = eigenvectors[:, sorted_indices][:, :n]

    def project_to_subspace(self, y):
        """
        Projects an observed output onto the subspace spanned by significant eigenvectors.

        Parameters:
        -----------
        y : torch.Tensor
            Observed output.

        Returns:
        --------
        torch.Tensor
            Potential inputs in the subspace.
        """
        return torch.mm(self.significant_eigenvectors, torch.mm(self.significant_eigenvectors.T, y))

    def project_to_input_space(self):
        """
        Projects back to the discrete input space to find representative points.

        Returns:
        --------
        torch.Tensor
            Representative points in the input space.
        """
        projections = torch.mm(self.kernel_matrix, self.significant_eigenvectors)
        most_aligned_indices = torch.argmax(torch.abs(projections), dim=0)
        return self.X_train[most_aligned_indices]


class MultiDiscreteGaussianProcess:
    """
    Implements a Gaussian Process model that can handle multiple discrete distributions
    for both input and output.

    Attributes:
        kernel_func (callable): A function that computes the covariance between inputs.
        mean_func (callable): A function that computes the mean of the GP for given inputs.
    """

    def __init__(self, kernel_func, mean_func=None):
        """
        Initializes the MultiDiscreteGaussianProcess.

        Args:
            kernel_func (callable): A function that computes the covariance matrix
                between inputs. It should have the signature:
                    kernel_func(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor
            mean_func (callable, optional): A function that computes the mean of the GP
                for given inputs. If not provided, defaults to zero mean. It should have
                the signature:
                    mean_func(X: torch.Tensor) -> torch.Tensor
        """
        self.kernel_func = kernel_func
        self.mean_func = mean_func if mean_func else lambda x: torch.zeros(x.shape[0])

    def predict_with_multi_discrete_input_output(self, X, P, Q):
        """
        Computes the expected outputs and their covariance given multiple discrete distributions
        over the inputs for multiple outputs.

        Underlying Mathematics:
            Given two discrete distributions P and Q over the same set of inputs X, the expected
            mean for the first output dimension is:
                E_P[f1(x)] = m1^T P
            Similarly, for the second output dimension:
                E_Q[f2(x)] = m2^T Q
            The covariance between these expectations can be written as:
                Cov_P,Q[f1(x), f2(x)] = P^T K12 Q - E_P[f1(x)] E_Q[f2(x)]
            Where K12 is the cross-covariance matrix between the outputs of the GP for
            the two output dimensions.

        Args:
            X (torch.Tensor): A tensor of shape [num_points, input_dim] representing the
                discrete input points.
            P (torch.Tensor): A vector of shape [num_points] representing the discrete
                distribution over the input for the first output.
            Q (torch.Tensor): A vector of shape [num_points] representing the discrete
                distribution over the input for the second output.

        Returns:
            tuple: A tuple containing:
                - A tuple of expected means for the two outputs.
                - The expected covariance between the two outputs.
        """
        # Compute mean and covariance matrix for the points in X
        m = self.mean_func(X)
        K = self.kernel_func(X, X)

        # Split mean and covariance for the two outputs
        m1, m2 = m[:, 0], m[:, 1]
        K11, K22 = K, K  # Assuming same kernel for both outputs for simplicity
        K12 = K  # Cross-covariance; for independent outputs, this would be zero

        # Expected means for the two outputs
        expected_mean1 = torch.dot(m1, P)
        expected_mean2 = torch.dot(m2, Q)

        # Covariance between the expectations
        expected_cov = torch.dot(P, torch.mv(K12, Q)) - expected_mean1 * expected_mean2

        return (expected_mean1, expected_mean2), expected_cov
