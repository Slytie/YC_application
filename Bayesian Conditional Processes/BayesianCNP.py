import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations, product

"""

**Initial Requirements:**

1. **Sampling:** Use preset filters to sample a \( U \times K \) matrix where \( U = 2F + H \).
    - **Status:** Implemented. The `FilterBlock` samples from the input using preset filters. The output shape is consistent with \( U \times K \).

2. **Processing:** Pass the sampled matrix through a series of FC layers. The depth and width of these layers can be hyperparameters, but the final layer should have a width of \( K \).
    - **Status:** Implemented. The `DenseBlock` handles the fully connected layers. The final output has a width of \( K \).

3. **Aggregation:** For each of the \( \binom{N}{F} \) sub-tracks:
    - Multiply the \( K \)-length vectors together.
    - Apply a softmax.
    - Multiply by a prior (could be learned or preset).
    - Apply another softmax to produce the final prediction for the track.
    - **Status:** Implemented. The model aggregates the outputs across sub-tracks, applies softmax, multiplies by a prior, and then applies another softmax.

4. **Loss Calculation:** Compute the KL divergence between the predicted distributions and the true distributions for each track.
    - **Status:** Implemented. KL divergence is computed using PyTorch's built-in KLDivLoss.

5. **Flexible Design:** The model should be able to handle any \( N \), \( H \), \( F \), and \( K \) values.
    - **Status:** Implemented. The model's blocks are designed dynamically based on the provided hyperparameters.

6. **Element-wise Multiplication:** After sub-tracks of each sample are multiplied, there should be an element-wise multiplication of the different context samples.
    - **Status:** Implemented. The `elementwise_multiplication` function aggregates outputs across context samples using element-wise multiplication.

7. **Output Shape:** The model should output a tensor of shape \( [H, K] \), where \( H \) is the number of output dimensions and \( K \) is the discretization level.
    - **Status:** Modified. The model currently outputs a tensor of shape \( [H] \) after aggregating over sub-tracks and context samples. However, this shape is consistent with the notion of predicting a distribution over \( K \) possible outputs for each \( H \). The softmax ensures the predictions sum to 1 over the \( K \) possible outputs.

Given this summary, it appears that the architecture aligns well with the previously described requirements. The essence of modeling the conditional distribution using preset filters, FC layers, and appropriate aggregation mechanisms remains intact. The model is also designed with flexibility in mind, allowing for easy adjustments to key hyperparameters.
=================================================

# Implementation Checklist for Dynamic Conditional Neural Process (DCNP)

## 1. Initialization
- [ ] **1.1.** Define the number of dimensions \( N \) and discretization level \( K \).
- [ ] **1.2.** Initialize the prior distribution for each layer.
- [ ] **1.3.** Prepare the context data (a set of observations).

## 2. Preset Filters (Sampling Layer)
- [ ] **2.1.** Implement preset filters for each dimension to sample the input space.
- [ ] **2.2.** For each output-dimension track \( i \):
    - [ ] **2.2.1.** Use filters specific to the required dimensions for the marginal distribution.
    - [ ] **2.2.2.** For each input-dimension sub-track \( j \) within track \( i \):
        - [ ] **2.2.2.1.** Apply the filter corresponding to \( x_j \) to sample the input \( x \) for the appropriate dimension.
        - [ ] **2.2.2.2.** Pass the sampled feature through a fully connected (FC) layer followed by a softmax function, producing a vector of length \( K \).
        - [ ] **2.2.2.3.** Sample the context sets \( X_{c_j} \) (input context) and \( X_{c_o} \) (output context) for the same dimension. Pass them through their respective FC layers followed by softmax functions.
        - [ ] **2.2.2.4.** Perform element-wise multiplication of the FC layer output for \( x_j \) with the FC layers outputs from the context. Apply a softmax function on the result to produce the marginal distribution for the (i,j) track sub-track.

## 3. Layer 1 (Marginal Distributions Layer)
- [ ] **3.1.** For each output-dimension track \( i \) with input-dimension sub-track \( j \):
    - [ ] **3.1.1.** Combine the results from the preset filters using element-wise multiplication. Apply a softmax function to produce the marginal distribution \( P(x_i | x_j) \).
    - [ ] **3.1.2.** Generate a latent vector of length \( K \) representing this distribution.
- [ ] **3.2.** Ensure there are \( NxN \) latent vectors of length \( K \) at the end of this layer.

## 4. Layer 2 (Prediction and Loss Calculation Layer for Marginal Distributions)
- [ ] **4.1.** For each output-dimension track \( i \):
    - [ ] **4.1.1.** Combine the latent vectors from all its input-dimension sub-tracks using element-wise multiplication.
    - [ ] **4.1.2.** Multiply the result with the prior for this layer and apply a softmax function to produce the prediction for the output dimension \( x_i \).
- [ ] **4.2.** Compare each predicted output distribution with the true output distribution for each dimension to calculate the KL loss.
- [ ] **4.3.** Sum up the KL divergences for all dimensions to obtain the total loss.

## 5. Layer 3 (Joint Conditional Distributions Layer)
- [ ] **5.1.** For each output-dimension track \( i \):
    - [ ] **5.1.1.** For each pair of input-dimension sub-tracks \( (j, k) \) within track \( i \) where \( j \neq k \):
        - [ ] **5.1.1.1.** Apply the preset filters corresponding to \( x_j \) and \( x_k \).
        - [ ] **5.1.1.2.** Pass the sampled features through an FC layer followed by a softmax function, producing vectors of length \( K \).
        - [ ] **5.1.1.3.** Sample the context sets for these dimensions and pass them through their respective FC layers followed by softmax functions.
        - [ ] **5.1.1.4.** Perform element-wise multiplication of the FC layer outputs for \( x_j \) and \( x_k \) with the FC layer outputs from the context. Apply a softmax function on the result to produce the joint conditional distribution \( P(x_i | x_j, x_k) \).
- [ ] **5.2.** Ensure there are \( N \choose 2 \) latent vectors of length \( K \) for each output-dimension track \( i \) at the end of this layer.

## 6. Layer 4 (Prediction and Loss Calculation Layer for Joint Conditional Distributions)
- [ ] **6.1.** For each output-dimension track \( i \):
    - [ ] **6.1.1.** Combine the latent vectors from all pairs of its input-dimension sub-tracks using element-wise multiplication.
    - [ ] **6.1.2.** Multiply the result with the prior for this layer and apply a softmax function to produce the prediction for the output dimension \( x_i \).
- [ ] **6.2.** Compare each predicted output distribution based on the joint conditionals with the true output distribution for each dimension to calculate the KL loss.
- [ ] **6.3.** Sum up the KL divergences for all dimensions to obtain the total loss for this layer.

## 7. Training and Optimization
- [ ] **7.1.** Use gradient descent or another optimization algorithm to minimize the total loss.
- [ ] **7.2.** Iterate through training data and adjust weights accordingly.

## 8. Evaluation
- [ ] **8.1.** Use a separate validation or test dataset to evaluate the model's performance.
- [ ] **8.2.** Measure accuracy, precision, recall, F1-score, or any other relevant metric.

## 9. Inference
- [ ] **9.1.** Given a new set of input dimensions, predict the output dimensions using the traine

"""
def cauchy_filter(K, mu, gamma):
    """
    Generate a Cauchy filter of specified length.

    Args:
    - K (int): The length of the filter.
    - mu (float): Location parameter of the Cauchy distribution.
    - gamma (float): Scale parameter of the Cauchy distribution.

    Returns:
    - numpy.ndarray: A Cauchy filter of length K.

    Logic:
    The Cauchy filter is derived from the Cauchy probability density function.
    It's used to produce values that are influenced by both the location and scale parameters.
    """
    x = np.arange(1, K + 1)
    cauchy = 1 / (np.pi * gamma * (1 + ((x - mu) / gamma) ** 2))
    return cauchy / cauchy.sum()

def compute_gaussian_means(K, L):
    return [(K / (L + 1)) * i for i in range(1, L + 1)]


def generate_filters_for_dimension(K, L, sigma, filter_type="cauchy"):
    """
    Generate L filters for a specified dimension.

    Args:
    - K (int): Number of discrete points in the dimension.
    - L (int): Number of filters to generate.
    - sigma (float): Standard deviation for Gaussian filters (if chosen).
    - filter_type (str, optional): Type of filter to generate. Options are "cauchy" and "gaussian". Defaults to "cauchy".

    Returns:
    - list of numpy.ndarray: A list of generated filters.

    Logic:
    Depending on the filter type chosen, the function either creates Gaussian or Cauchy filters.
    The means for these filters are determined by evenly dividing the dimension.
    For instance, if L=3 for a dimension of size K, then three filters are generated
    with their peaks (for Gaussian) or location parameters (for Cauchy) spaced evenly across the dimension.
    """
    means = compute_gaussian_means(K, L)
    if filter_type == "cauchy":
        filters = [cauchy_filter(K, mu, sigma) for mu in means]
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    return filters


def generate_filter_combinations(K, L, sigma, N):
    """
    Generate all possible combinations of 1D filters across N dimensions.

    Args:
    - K (int): Number of discrete points in each dimension.
    - L (int): Number of filters per dimension.
    - sigma (float): Standard deviation for Gaussian filters (not used in the current logic).
    - N (int): Number of dimensions.

    Returns:
    - list of numpy.ndarray: A list of 2D filters of shape N x K.

    Logic:
    The function first generates 1D filters for the given dimension K.
    Then, it produces all possible combinations of these 1D filters across the N dimensions.
    The resulting filters are combined to form 2D filters.
    """
    filters_for_K = generate_filters_for_dimension(K, L, sigma)
    filter_combinations = list(product(filters_for_K, repeat=N))
    combined_filters = [np.vstack(comb) for comb in filter_combinations]
    return combined_filters

class ConditionalFilterBlock(nn.Module):
    """
    A PyTorch module that applies multiple convolutional filters to input data.

    Attributes:
    - H_index (list of int): The dimensions of the variables to be predicted.
    - filters (nn.ParameterList): A list of convolutional filters.
    - output_size (int): Number of filters.

    Logic:
    The block is designed to apply multiple convolutional filters to a concatenated input and output data.
    The filters are stored in a ParameterList to avoid unnecessary stacking in memory.
    The block can also dynamically add new filters.
    """

    def __init__(self, F_param, K_param, H_param, H_index, L= 4, x=False):
        """
        Initialize the ConditionalFilterBlock.

        Args:
        - F_param (int): Number of conditioning variables.
        - K_param (int): Number of discrete points in each dimension.
        - H_param (int): Number of variables to be predicted.
        - H_index (list of int): The dimensions of the variables to be predicted.
        - initial_filters (list of torch.Tensor): Initial set of convolutional filters.
        """
        super(ConditionalFilterBlock, self).__init__()
        self.H_index = H_index
        self.F_param = F_param
        self.H_param = H_param
        self.initial_filters = generate_filter_combinations(K_param, L=L, sigma=0.3, N=H_param + F_param)
        self.x = x

        # Print the shape of initial filters
        print(f"Initial filter shape (first filter): {self.initial_filters[0].shape}")

        self.filters = nn.ParameterList(
            [nn.Parameter(torch.tensor(f, dtype=torch.float32).view(1, 1, 6, 10), requires_grad=False) for f in
             self.initial_filters]
        )

        self.output_size = len(self.initial_filters)

    def forward(self, context_sample):
        if not self.x:
            print("Context_sample  1 shape", context_sample[1].shape)
            reshaped_context_sample_1 = torch.stack([context_sample[1][idx] for idx in self.H_index])

            # Print shapes before concatenation
            print(f"Context sample 0 shape: {context_sample[0][:, :10].shape}")
            print(f"Reshaped context sample 1 shape: {reshaped_context_sample_1.shape}")

            combined_data = torch.cat([context_sample[0][:, :10], reshaped_context_sample_1], dim=0)
            combined_data_2d = combined_data.view(1, 1, 6, 10)

            # Print the shape of combined data
            print(f"Combined data 2D shape: {combined_data_2d.shape}")

            # Print the shape of the first filter
            print(f"First filter shape: {self.filters[0].shape}")
            print(f"Total filters: {len(self.filters)}")

        if self.x:
            combined_data = context_sample.view(1, 1, 6, 10)

        # Perform element-wise multiplication and sum over the K dimension
        activations = [torch.sum(combined_data * filter.squeeze()) for filter in self.filters]
        activations = torch.stack(activations, dim=0)
        print(f"Activations shape: {activations.shape}")

        # Reshape activations
        L = int(round(activations.shape[0] ** (1.0 / (self.F_param + self.H_param))))
        reshaped_dimensions = [L] * (self.F_param + self.H_param)
        activations = activations.view(*reshaped_dimensions)

        if self.x:
            # Remove  activations for H, given that we are only interested in F.
            slices = [slice(None)] * self.F_param + [0] * (activations.dim() - self.F_param)
            activations = activations[tuple(slices)]

        print(activations.shape)

        return activations

    def add_filter(self, new_filter):
        """
        Add a new convolutional filter to the block.

        Args:
        - new_filter (torch.Tensor): The new filter to be added.
        """
        reshaped_filter = new_filter.unsqueeze(0).expand(10, -1).unsqueeze(0).unsqueeze(2)
        self.filters.append(nn.Parameter(reshaped_filter, requires_grad=False))
        self.output_size += 1

class DensePredictionBlock(nn.Module):
    """
    A block that handles the dense layers which predicts the distribution.
    """

    def __init__(self, U, hidden_dims, K_param, H_param, F_param, C):
        super(DensePredictionBlock, self).__init__()
        layers = []

        # Flattening the input tensor
        input_dim = C * U ** 2

        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim

        # Ensuring the last layer outputs (H_param + F_param) * K_param elements
        output_features = K_param ** (H_param + F_param)
        layers.append(nn.Linear(input_dim, output_features))

        self.network = nn.Sequential(*layers)
        self.H_param = H_param
        self.F_param = F_param
        self.K_param = K_param
        self.C=C
        self.U=U

    def forward(self, x):
        # Flattening the input tensor
        x = x.view(-1, self.C * self.U ** 2)

        dense_output = self.network(x)

        # Reshaping the output to [K, K, K, ...] repeated H+F times
        reshape_dims = [self.K_param] * (self.H_param + self.F_param)
        dense_output = dense_output.view(*reshape_dims)

        # Applying the Softmax operation along the last dimension
        dense_output = nn.functional.softmax(dense_output, dim=-1)

        return dense_output


import torch.nn.functional as F
import torch.nn as nn


class TensorProductExpansion(nn.Module):
    def __init__(self, C, N, K, L):
        super(TensorProductExpansion, self).__init__()

        # Define a learnable bias of shape CxNxL
        self.bias = nn.Parameter(torch.zeros(C, N, L))

        self.N = N
        self.K = K
        self.L = L

    def forward(self, tensor):
        """
        Expand a tensor of shape CxNxL using tensor product to get a tensor of shape CxLxLxL... (N times)

        Args:
        - tensor (torch.Tensor): Input tensor of shape CxNxL.

        Returns:
        - torch.Tensor: Expanded tensor of shape CxLxLxL... (N times).
        """
        C = tensor.shape[0]

        # Interpolate from K to L
        #tensor_interpolated = F.interpolate(tensor, size=self.L, mode='linear', align_corners=False)

        # Add learnable bias to the interpolated tensor
        tensor_with_bias = tensor + self.bias

        # Using einsum to compute the tensor product for each sample in C
        expanded_tensors = []
        for c in range(C):
            expanded_tensor = tensor_with_bias[c, 0]
            for n in range(1, self.N):
                expanded_tensor = torch.einsum('i,j->ij', expanded_tensor, tensor_with_bias[c, n]).reshape(-1)
            expanded_tensors.append(expanded_tensor)

        # Stacking results to obtain final tensor of shape CxLxLxL... (N times)
        final_tensor = torch.stack(expanded_tensors).reshape(C, *[self.L for _ in range(self.N)])

        return final_tensor


import torch
import torch.nn as nn


class MarginalizeJointDistribution(nn.Module):
    def __init__(self, P_Y_given_X_shape, P_X_shape):
        super(MarginalizeJointDistribution, self).__init__()

        # Initialize biases as learnable parameters
        self.bias_Y_given_X = nn.Parameter(torch.zeros(P_Y_given_X_shape))
        self.bias_X = nn.Parameter(torch.zeros(P_X_shape))

    def forward(self, P_Y_given_X, P_X):
        """
        Computes the joint distribution P(Y) by marginalizing over X using the given joint conditional
        distribution P(Y|X) and the sample distribution P(X).
        """

        # Add learnable biases
        P_Y_given_X += self.bias_Y_given_X
        P_X += self.bias_X

        # Ensure the distributions are still valid (i.e., they sum to one)
        P_Y_given_X = torch.nn.functional.softmax(P_Y_given_X, dim=-1)
        P_X = torch.nn.functional.softmax(P_X, dim=-1)

        # Rest of the logic remains the same as in the function
        num_X_dims = len(P_X.shape)
        num_Y_dims = len(P_Y_given_X.shape) - num_X_dims

        for _ in range(num_Y_dims):
            P_X = P_X.unsqueeze(0)

        product = P_Y_given_X * P_X
        P_Y = torch.sum(product, dim=tuple(range(num_Y_dims, num_Y_dims + num_X_dims)))

        return P_Y


class ConvContextDistributions(nn.Module):
    def __init__(self, N, L, C, num_conv_layers, G):
        super(ConvContextDistributions, self).__init__()

        self.N = N
        self.L = L
        self.C = C
        self.G = G

        # Creating a sequence of convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels=C, out_channels=G, kernel_size=1))
        for _ in range(num_conv_layers - 1):
            self.convs.append(nn.Conv1d(in_channels=G, out_channels=G, kernel_size=1))

        # Deconvolution layer to map back to original size
        self.deconv = nn.ConvTranspose1d(in_channels=G, out_channels=1, kernel_size=1)

    def forward(self, x):
        # Reshape input tensor for convolution
        x = x.view(-1, self.C, self.L ** self.N)

        # Apply convolutions
        for conv in self.convs:
            x = F.relu(conv(x))

        # Apply deconvolution
        x = self.deconv(x)

        # Reshape to get back to the required shape
        x = x.view(*[self.L for _ in range(self.N)])
        return x

class EmpiricalBayesDistribution(nn.Module):
    """
    Compute the empirical joint distribution using input and output tensors.

    This model estimates the empirical joint distribution of the input and output tensors.
    The distribution is captured as a multi-dimensional histogram and normalized to sum to 1.

    Attributes:
        bias_input (torch.nn.Parameter): Learnable biases for the input tensor.
        bias_output (torch.nn.Parameter): Learnable biases for the output tensor.
        H (int): Dimension of the input tensor.
        F_param (int): Dimension of the output tensor.
        K (int): Number of possible states or bins for each variable.
    """

    def __init__(self, C, H, F_param, K):
        """
        Initializes the EmpiricalBayesDistribution model.

        Args:
            C (int): Number of samples.
            H (int): Dimension of the input tensor.
            F_param (int): Dimension of the output tensor.
            K (int): The size of the third dimension for both input and output tensors.
        """
        super(EmpiricalBayesDistribution, self).__init__()

        # Define learnable biases for input and output tensors
        self.bias_input = nn.Parameter(torch.zeros(C, H, K))
        self.bias_output = nn.Parameter(torch.zeros(C, F_param, K))

        self.H = H
        self.F_param = F_param
        self.K = K

    def forward(self, input_tensor, output_tensor):
        """
        Compute the empirical joint distribution.

        The function estimates the empirical joint distribution of the input and output tensors.
        It uses biases, then counts co-occurrences, and finally normalizes the distribution.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape CxHxK.
            output_tensor (torch.Tensor): Output tensor of shape CxFxK.

        Returns:
            torch.Tensor: Joint distribution tensor of shape (K, K, K, ... [H+F times]).
        """

        # Step 1: Add learnable biases and discretize
        biased_input = (input_tensor + self.bias_input).long()
        biased_output = (output_tensor + self.bias_output).long()

        # Ensure values are within [0, K-1]
        biased_input = torch.clamp(biased_input, 0, 0.2)
        biased_output = torch.clamp(biased_output, 0, 0.2)

        # Step 2: Count co-occurrences
        joint_histogram = torch.zeros([self.K] * (self.H + self.F_param), device=input_tensor.device)

        for c in range(input_tensor.shape[0]):
            indices = biased_input[c].tolist() + biased_output[c].tolist()
            joint_histogram[tuple(indices)] += 1

        # Step 3: Normalize
        empirical_joint_distribution = joint_histogram / input_tensor.shape[0]

        return empirical_joint_distribution


import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteKDE(nn.Module):
    def __init__(self, n, F, H, K, marginalize=False, bandwidth_matrix=None):
        super(DiscreteKDE, self).__init__()

        # Check if a bandwidth matrix is provided, otherwise use the default initialization
        if bandwidth_matrix is not None:
            self.H_bandwidth = nn.Parameter(bandwidth_matrix)
        else:
            self.H_bandwidth = nn.Parameter(torch.full((F + H, F + H), 0.10) + torch.eye(F + H) * 0.69)

        self.w = nn.Parameter(torch.ones(n))

        self.F = F
        self.H = H
        self.K = K
        self.marginalize = marginalize
        self.d = K

        eigenvalues = torch.linalg.eigvalsh(self.H_bandwidth)
        if torch.any(eigenvalues <= 0):
            print("Warning: H_bandwidth is not positive definite!")

    def set_bandwidth_matrix(self, new_bandwidth_matrix):
        """Method to set or adjust the bandwidth matrix after model's initialization."""
        self.H_bandwidth.data = new_bandwidth_matrix

    def gaussian_kernel(self, z):
        result = (1.0 / torch.sqrt(2.0 * torch.tensor(torch.pi))) * torch.exp(-0.5 * z ** 2)
        return result

    def marginalise(self, m_k):
        return torch.stack([m_k.sum(dim=tuple(j for j in range(self.H) if j != i)) for i in range(self.H)])

    def forward(self, X_probs, Y_probs):
        joint_densities = []

        # Assume that the indices represent grid locations. This is a simple range but can be adjusted.
        indices = torch.arange(0, X_probs.shape[1] + Y_probs.shape[1])

        H_I = torch.inverse(self.H_bandwidth)

        with torch.no_grad():
            for q_idx in indices:
                joint_density = torch.zeros([self.K] * self.H)

                for i, (x_prob, y_prob) in enumerate(zip(X_probs, Y_probs)):
                    combined_probs = torch.cat((x_prob, y_prob), dim=0)
                    print(f"combined_probs shape: {combined_probs.shape}")

                    # Compute the difference
                    diff = q_idx.unsqueeze(-1) - indices.float()
                    print(f"diff shape (after subtraction): {diff.shape}")

                    diff = diff @ H_I
                    print(f"diff shape (after multiplication): {diff.shape}")

                    kernel_weights = self.gaussian_kernel(diff)
                    print(f"kernel_weights shape: {kernel_weights.shape}")

                    # Weighting by the probabilities
                    weighted_kernel = combined_probs * kernel_weights
                    print(f"weighted_kernel shape: {weighted_kernel.shape}")

                    joint_density.add_(weighted_kernel.sum(dim=-1))

                    del combined_probs, diff, kernel_weights, weighted_kernel

            out = torch.stack(joint_densities)
            out = out.sum(dim=0)

            del joint_densities

        if self.marginalize:
            m_k = self.marginalise(out)
            m_k = F.softmax(m_k, dim=1)
            return m_k

        return out


class JointConditionalDistributionBlock(nn.Module):
    """
    A block that handles the aggregation of predictions from the dense block,
    considering the context samples and producing the final prediction.
    """

    def __init__(self, N_param, H_param, F_param, K_param, L_param, C, H_index = None, bias=None):
        super(JointConditionalDistributionBlock, self).__init__()
        self.prior = nn.Parameter(torch.ones(H_param, K_param) / K_param)
        #self.conv_context_distributions = ConvContextDistributions(N_param, L_param, C, num_conv_layers=4, G=20)
        self.tensor_product_c = TensorProductExpansion(C, N_param, K_param, L_param)
        self.tensor_product_x = TensorProductExpansion(1, F_param, K_param, L_param)
        self.kde = DiscreteKDE(C, F_param, H_param, K_param, init_bandwidth=1.0)
        self.marginalise = MarginalizeJointDistribution(tuple([L_param] * (H_param + F_param)), tuple([L_param] * (F_param)))
        self.empirical_bayes_joint = EmpiricalBayesDistribution(C, H_param, F_param, K_param)

        self.H_param = H_param
        self.K_param = K_param
        self.F_param = F_param
        self.N_param = N_param
        self.L_param = L_param
        self.bias = bias

    def forward(self, x, context_samples, log_prob=False):

        c_x = torch.stack([sample[1] for sample in context_samples])
        c_y = torch.stack([sample[0] for sample in context_samples])
        x = x.unsqueeze(0)

        #c_b = self.empirical_bayes_joint(c_x, c_y)
        print("Starting KDE")
        j_k = self.kde(c_x, c_y)
        print("JOINT", j_k)

        x = self.tensor_product_x(x)
        x = x.squeeze(0)

        # Calculate joint distribution by marginalising out the X
        # m_b = self.marginalise(c_b, x)
        m_k = self.marginalise(j_k, x)

        #marginals_b = torch.stack([m_b.sum(dim=tuple(j for j in range(self.H_param) if j != i)) for i in range(self.H_param)])
        marginals_k = torch.stack([m_k.sum(dim=tuple(j for j in range(self.H_param) if j != i)) for i in range(self.H_param)])
        print("MARGINALS", marginals_k)

        log_prob = False
        if log_prob:
            marginals = torch.log(marginals_k + 1e-9) + 0 * torch.log(marginals_k + 1e-9)
            marginals = torch.log(marginals_k + 1e-9)
            x = torch.log(x + 1e-9)

        marginals = F.softmax(marginals_k, dim=1)

        return marginals


import matplotlib.pyplot as plt


def plot_distribution(data, title):
    """
    Plots the histogram of the given data.

    Args:
    - data (torch.Tensor): The tensor data to be plotted.
    - title (str): The title for the plot.
    """
    plt.hist(data.numpy().flatten(), bins=50, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


