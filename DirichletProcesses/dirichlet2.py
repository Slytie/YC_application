import torch


class DirichletProcessDiscrete:
    """
    A class to model a Dirichlet Process in a discrete space.

    Attributes:
        H (int): Dimension of the input tensor.
        F (int): Dimension of the conditioning variables.
        K (int): Size of the third dimension for both input and output tensors.
    """

    def __init__(self, H, F, K):
        """
        Initializes DirichletProcess model.

        Args:
            H (int): Dimension of the input tensor.
            F (int): Dimension of the conditioning variables.
            K (int): The size of the third dimension for both input and output tensors.
        """
        self.H = H
        self.F = F
        self.K = K

    def stick_breaking(self, alpha, n_components=100):
        """
        Generate weights using the stick-breaking process.

        Args:
            alpha (float): Concentration parameter for the Dirichlet Process.
            n_components (int, optional): Number of components. Defaults to 100.

        Returns:
            torch.Tensor: Weights generated by the stick-breaking process.
        """
        v = torch.distributions.Beta(1, alpha).sample([n_components])
        pi = torch.empty_like(v)
        pi[0] = v[0]
        for i in range(1, n_components):
            pi[i] = v[i] * torch.prod(1 - v[:i])
        pi /= pi.sum()
        return pi

    def assign_to_component(self, data, weights):
        """
        Assign data points to components.

        Args:
            data (torch.Tensor): Quantized data points.
            weights (torch.Tensor): Weights from the stick-breaking process.

        Returns:
            torch.Tensor: Component assignments for each data point.
        """
        n_components = weights.shape[0]
        likelihood = weights.unsqueeze(0).repeat(data.shape[0], 1)
        component_assignment = torch.argmax(likelihood, dim=1)
        return component_assignment

    def compute_centroids(self, data, assignments, n_components):
        """
        Compute centroids for the given component assignments.

        Args:
            data (torch.Tensor): Data points.
            assignments (torch.Tensor): Component assignments for each data point.
            n_components (int): Number of components or clusters.

        Returns:
            torch.Tensor: Centroids for each component.
        """
        centroids = []
        for i in range(n_components):
            component_data = data[assignments == i]
            if component_data.shape[0] > 0:
                centroid = component_data.mean(dim=0)
            else:
                centroid = torch.zeros(data.shape[1])
            centroids.append(centroid)
        return torch.stack(centroids)

    def assign_based_on_distance(self, data, centroids):
        """
        Assign each data point to the component with the nearest centroid.

        Args:
            data (torch.Tensor): Data points.
            centroids (torch.Tensor): Centroids for each component.

        Returns:
            torch.Tensor: Component assignments for each data point.
        """
        distances = torch.cdist(data, centroids)
        assignments = torch.argmin(distances, dim=1)
        return assignments

    def quantize(self, data, K=10):
        """
        Quantize the data into K discrete values along each dimension.

        Args:
            data (torch.Tensor): Data points.
            K (int, optional): Number of bins for quantization. Defaults to 10.

        Returns:
            torch.Tensor: Quantized data.
        """
        min_vals = torch.min(data, dim=0).values
        max_vals = torch.max(data, dim=0).values

        bin_indices_list = []
        for i in range(data.shape[1]):
            bin_edges = torch.linspace(min_vals[i], max_vals[i], K + 1)
            bin_indices_dim = torch.bucketize(data[:, i], bin_edges)
            bin_indices_dim = torch.clamp(bin_indices_dim - 1, 0, K - 1)
            bin_indices_list.append(bin_indices_dim.unsqueeze(-1))
        bin_indices = torch.cat(bin_indices_list, dim=1)
        return bin_indices


# Example usage:
dp = DirichletProcessDiscrete(H=3, F=2, K=10)
data = torch.randn(100, 5)  # Sample data with 5 dimensions
quantized_data = dp.quantize(data)
weights = dp.stick_breaking(alpha=2.0)
assignments = dp.assign_to_component(quantized_data, weights)
centroids = dp.compute_centroids(data, assignments, len(weights))
new_assignments = dp.assign_based_on_distance(data, centroids)
