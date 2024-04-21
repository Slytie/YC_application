# Compile the entire corrected script
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import random

def generate_N_lookup_table(S, N):
    """
    Generate a lookup table for numbers based on N multipliers and numbers from 1 to S.
    """
    table = {}
    for i in range(1, S + 1):
        table[i] = tuple(i + j * S for j in range(1, N + 1))
    return table

def generate_pairwise_lookup_table_corrected(S, N, X=10):
    """
    Generate a pairwise lookup table based on S, N and X with values ranging from S*N to S*N + X.
    """
    original_lookup = generate_N_lookup_table(S, N)
    pairwise_table = {}

    for key, values in original_lookup.items():
        for val in values:
            pair = (key, val)
            pairwise_value = random.randint(S * N, S * N + X)
            pairwise_table[pair] = pairwise_value

    return pairwise_table

# Function to generate a list of numbers from 1 to S
def generate_list_of_numbers(S):
    return list(range(1, S + 1))

# Function to generate a lookup table
def generate_lookup(S, N):
    numbers = list(range(S + 1, S * N + 1))
    lookup = {}
    for i in range(1, S + 1):
        lookup[i] = []
        for _ in range(N):
            value = numbers.pop(0)
            lookup[i].append(value)
            numbers.append(value)
    return lookup

# Function to generate N numbers based on a given number and a lookup table
def get_N_numbers(number, lookup, N):
    return lookup[number][:N]

# Modify the numpy array generation function in the second script to use the pairwise lookup table
def generate_numpy_array_synchronized_corrected(S, N, X=2):
    original_lookup = generate_N_lookup_table(S, N)
    pairwise_lookup = generate_pairwise_lookup_table_corrected(S, N, X)

    # Initialize a blank numpy array
    array = np.zeros((S, N, N), dtype=int)

    # Populate the i values
    for i in range(1, S + 1):
        array[i - 1, :, 0] = i

    # Populate the j and k values using the consistent pairwise lookup table
    for (i, j), k in pairwise_lookup.items():
        j_idx = original_lookup[i].index(j)
        array[i - 1, j_idx, 1] = j
        array[i - 1, j_idx, 2] = k

    return array, pairwise_lookup

# Function to generate a sparse transition matrix
def generate_sparse_transition_matrix(S):
    transition_matrix = np.zeros((S, S))
    for i in range(S):
        available_indices = list(range(S))
        available_indices.remove(i)
        chosen_index = np.random.choice(available_indices)
        transition_matrix[i, chosen_index] = 1
    return transition_matrix

# Function to normalize a matrix
def normalize_matrix(matrix):
    row_sums = matrix.sum(axis=1)
    normalized_matrix = matrix / row_sums[:, np.newaxis]
    return normalized_matrix

# Custom function to select an index based on the given probability distribution
def custom_choice(probabilities):
    cumulative_probs = np.cumsum(probabilities)
    random_number = np.random.rand()
    selected_index = np.searchsorted(cumulative_probs, random_number)
    return selected_index


def apply_transition_matrix_with_custom_choice(original_array, transition_matrix, pairwise_lookup):
    S, N, _ = original_array.shape
    new_array = np.zeros((S, N, N), dtype=int)

    # Generate the original lookup table for the correct value of S
    original_lookup = generate_N_lookup_table(S, N)

    for idx in range(S):
        # Transition for i values
        i_value = custom_choice(transition_matrix[idx, :])
        new_array[idx, :, 0] = i_value + 1

    for i in range(S):
        # Transition for j values using the original lookup table
        j_values = original_lookup[new_array[i, 0, 0]]
        for jdx, j in enumerate(j_values):
            new_array[i, jdx, 1] = j
            # Transition for k values using the pairwise lookup table
            new_array[i, jdx, 2] = pairwise_lookup[(new_array[i, jdx, 0], new_array[i, jdx, 1])]

    return new_array


def generate_M_samples(M, S, N):
    """
    Generate M samples of the array_synchronized (input) and new_array (output).
    """
    original_lookup = generate_N_lookup_table(S, N)
    input_samples = []
    output_samples = []

    for _ in range(M):
        array_synchronized, original_lookup = generate_numpy_array_synchronized_corrected(S, N)
        sparse_transition_matrix = generate_sparse_transition_matrix(S)
        normalized_transition_matrix = normalize_matrix(sparse_transition_matrix)
        new_array = apply_transition_matrix_with_custom_choice(array_synchronized, normalized_transition_matrix,
                                                               original_lookup)

        input_samples.append(array_synchronized)
        output_samples.append(new_array)

    # Stack the samples together
    input_array = np.stack(input_samples)
    output_array = np.stack(output_samples)

    return input_array, output_array




def create_data_loaders(input_array, output_array, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    """
    Create DataLoader instances for training, validation, and test sets.

    Parameters:
    - input_array: Input data in the shape MxSxNxN.
    - output_array: Output data in the shape MxSxNxN.
    - batch_size: Batch size for the DataLoader.
    - train_ratio: Ratio of samples to be used for training.
    - val_ratio: Ratio of samples to be used for validation.

    Returns:
    - train_loader, val_loader, test_loader: DataLoader instances for training, validation, and test sets respectively.
    """
    print(f"Highest value in the input array: {output_array.max()}")


    # Split the data based on the specified ratios
    num_samples = input_array.shape[0]

    # Indices for splitting
    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)

    train_input, train_output = input_array[:train_end], output_array[:train_end]
    val_input, val_output = input_array[train_end:val_end], output_array[train_end:val_end]
    test_input, test_output = input_array[val_end:], output_array[val_end:]

    # Convert to PyTorch tensors
    train_input, train_output = torch.from_numpy(train_input), torch.from_numpy(train_output)
    val_input, val_output = torch.from_numpy(val_input), torch.from_numpy(val_output)
    test_input, test_output = torch.from_numpy(test_input), torch.from_numpy(test_output)

    # Create DataLoader instances
    train_loader = DataLoader(list(zip(train_input, train_output)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(val_input, val_output)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(test_input, test_output)), batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

