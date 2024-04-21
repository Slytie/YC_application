import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class EMDist:
    def __init__(self, N, M, P, S):
        """
        Initialize the EMDistDocumented class with the given parameters.

        Args:
            N (int): Number of spaces.
            M (int): Number of clusters per space.
            P (int): Number of dimensions per cluster.
            S (int): Number of samples per cluster.
        """
        (N=4, M=5, P=3, S=50)

        self.N = N
        self.M = M
        self.P = P
        self.S = S

        # Initialize data structures to store results
        self.data_spaces = None
        self.indexed_data_spaces = None
        self.transition_matrix = None
        self.transitioned_data_spaces = None
        self.formatted_dataset = None

    def generate_data(self):
        """
        Generate synthetic data based on the defined constants.
        Populates the data_spaces attribute with generated data.
        """
        data_spaces = []

        for _ in range(self.N):
            space_data = []
            for _ in range(self.M):
                centroid = np.random.rand(self.P) * 10
                cluster_data = np.random.randn(self.S, self.P) + centroid
                distance_ids = np.abs(cluster_data - centroid)
                uncertainty_ids = np.random.rand(self.S, self.P)
                space_data.append((cluster_data, distance_ids, uncertainty_ids))

            data_spaces.append(space_data)

        self.data_spaces = data_spaces

    def index_mapping(self):
        """
        Convert raw values in the data to indices using dictionaries and bins.
        Ensures that each index is unique across clusters, dimensions, and distances.
        Populates the indexed_data_spaces attribute with indexed data.
        """
        # Calculate offsets for unique indexing
        max_cluster_index = self.N * self.M
        dimension_offset = max_cluster_index
        distance_offset = dimension_offset + self.P
        uncertainty_offset = distance_offset

        # Create dictionaries with offsets to ensure uniqueness
        unique_cluster_id_dict = {f"space_{i}_cluster_{j}": idx + 1 for idx, (i, j) in
                                  enumerate([(x, y) for x in range(self.N) for y in range(self.M)])}
        dimension_id_dict = {f"dimension_{i}": i + dimension_offset + 1 for i in range(self.P)}

        def value_to_bin(value, bins):
            return np.digitize(value, bins)

        distance_bins = np.linspace(0, np.max([np.max(cluster[1]) for space in self.data_spaces for cluster in space]),
                                    self.M) + distance_offset
        uncertainty_bins = np.linspace(0, 1, self.M) + uncertainty_offset

        indexed_data_spaces = []

        for space in self.data_spaces:
            indexed_space = []
            for cluster_data, distance_ids, uncertainty_ids in space:
                indexed_distance_ids = np.apply_along_axis(value_to_bin, 0, distance_ids, distance_bins)
                indexed_uncertainty_ids = np.apply_along_axis(value_to_bin, 0, uncertainty_ids, uncertainty_bins)
                indexed_space.append((cluster_data, indexed_distance_ids, indexed_uncertainty_ids))

            indexed_data_spaces.append(indexed_space)

        self.indexed_data_spaces = indexed_data_spaces

    def index_mapping(self):
        """
        Convert raw values in the data to indices using dictionaries and bins, ensuring unique indexing.
        Additionally, store the input dataset before transitioning.
        """
        # Calculate offsets for unique indexing
        max_cluster_index = self.N * self.M
        dimension_offset = max_cluster_index
        distance_offset = dimension_offset + self.P
        uncertainty_offset = distance_offset

        # Create dictionaries with offsets to ensure uniqueness
        unique_cluster_id_dict = {f"space_{i}_cluster_{j}": idx + 1 for idx, (i, j) in
                                  enumerate([(x, y) for x in range(self.N) for y in range(self.M)])}
        dimension_id_dict = {f"dimension_{i}": i + dimension_offset + 1 for i in range(self.P)}

        def value_to_bin(value, bins):
            return np.digitize(value, bins)

        distance_bins = np.linspace(0, np.max([np.max(cluster[1]) for space in self.data_spaces for cluster in space]),
                                    self.M) + distance_offset
        uncertainty_bins = np.linspace(0, 1, self.M) + uncertainty_offset

        indexed_data_spaces = []

        for space in self.data_spaces:
            indexed_space = []
            for cluster_data, distance_ids, uncertainty_ids in space:
                indexed_distance_ids = np.apply_along_axis(value_to_bin, 0, distance_ids, distance_bins)
                indexed_uncertainty_ids = np.apply_along_axis(value_to_bin, 0, uncertainty_ids, uncertainty_bins)
                indexed_space.append((cluster_data, indexed_distance_ids, indexed_uncertainty_ids))

            indexed_data_spaces.append(indexed_space)

        self.indexed_data_spaces = indexed_data_spaces

        # Create the input dataset with shape (sample_num, cluster_id, dimension_id, distance_id)
        input_dataset = np.zeros((self.S, self.N * self.M, self.P, self.M), dtype=int)

        for s in range(self.S):
            for n in range(self.N):
                for m in range(self.M):
                    cluster_data, distance_ids, _ = self.indexed_data_spaces[n][m]
                    unique_cluster_id = n * self.M + m
                    for p in range(self.P):
                        input_dataset[s, unique_cluster_id, p, distance_ids[s][p]] = 1

        self.input_dataset = input_dataset

    def get_input_dataset(self):
        """
        Retrieve the input dataset.
        """
        return self.input_dataset

    def get_output_dataset(self):
        """
        Retrieve the output dataset.
        """
        return self.formatted_dataset

    def initialize_transition_matrix(self):
        """
        Initialize and normalize the transition matrix.
        Populates the transition_matrix attribute with the generated matrix.
        """
        transition_matrix = np.random.rand(self.M, self.N)

        # Make the matrix sparser by setting values below a threshold to zero
        threshold = 0.4  # Adjust this value as needed
        transition_matrix[transition_matrix < threshold] = 0

        # Normalize the matrix
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

        self.transition_matrix = transition_matrix

    def transition_data(self):
        """
        Use the transition matrix to determine the output spaces for each data point.
        Populates the transitioned_data_spaces attribute with transitioned data.
        """
        transitioned_data_spaces = []

        for space_idx, space in enumerate(self.indexed_data_spaces):
            transitioned_space = []
            for cluster_idx, (cluster_data, distance_ids, uncertainty_ids) in enumerate(space):
                output_spaces = np.random.choice(self.N, size=cluster_data.shape[0],
                                                 p=self.transition_matrix[cluster_idx])
                transitioned_space.append((output_spaces, distance_ids, uncertainty_ids))

            transitioned_data_spaces.append(transitioned_space)

        self.transitioned_data_spaces = transitioned_data_spaces

    def format_data(self):
        """
        Format the transitioned data into the structure:
        (sample_num, cluster_id, dimension_id, distance_id).

        Populates the formatted_dataset attribute with the final dataset.
        """
        # New format shape
        formatted_dataset = np.zeros((self.S, self.N * self.M, self.P, self.M), dtype=int)

        for s in range(self.S):
            for n in range(self.N):
                for m in range(self.M):
                    cluster_data, distance_ids, uncertainty_ids = self.transitioned_data_spaces[n][m]
                    # Using the combination of space_id and cluster_id for unique cluster indexing
                    unique_cluster_id = n * self.M + m
                    for p in range(self.P):
                        formatted_dataset[s, unique_cluster_id, p, distance_ids[s][p]] = 1

        self.formatted_dataset = formatted_dataset

    def manual_split_data(self, train_size=0.8, val_size=0.1):
        assert (train_size + val_size) < 1.0, "Train and validation sizes combined should be less than 1.0"
        total_samples = len(self.formatted_dataset)
        train_samples = int(total_samples * train_size)
        val_samples = int(total_samples * val_size)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        self.train_data = self.formatted_dataset[indices[:train_samples]]
        self.val_data = self.formatted_dataset[indices[train_samples:train_samples+val_samples]]
        self.test_data = self.formatted_dataset[indices[train_samples+val_samples:]]


    def run(self):
        """
        Execute all the steps in the correct sequence:
        1. Data generation
        2. Index mapping
        3. Transition matrix initialization
        4. Data transitioning
        5. Data formatting

        After execution, the formatted_dataset attribute contains the final dataset.
        """
        self.generate_data()
        self.index_mapping()
        self.initialize_transition_matrix()
        self.transition_data()
        self.format_data()


class EMDistDataset(EMDist, Dataset):

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.S

    def __getitem__(self, idx):
        """
        Retrieve a specific sample (input-output pair) based on an index.

        Args:
        - idx: Index of the sample to retrieve

        Returns:
        - input_sample: Input data sample at the given index
        - output_sample: Output data sample at the given index
        """
        input_sample = self.input_dataset[idx]
        output_sample = self.formatted_dataset[idx]
        return torch.tensor(input_sample, dtype=torch.float32), torch.tensor(output_sample, dtype=torch.float32)


import torch
import matplotlib.pyplot as plt






# Create a DataLoader from the dataset
emdist_dataset = EMDistDataset(N=4, M=5, P=3, S=50)
emdist_dataset.run()
emdist_dataset.manual_split_data()
print(emdist_dataset.train_data.shape)

import numpy as np
from collections import defaultdict


def frequency_ndarray(arr):
    freq = defaultdict(int)

    # Flatten the ndarray to iterate through all elements
    flat_arr = arr.flatten()

    for val in flat_arr:
        freq[val] += 1

    return freq

fre=frequency_ndarray(emdist_dataset.train_data)
print(fre)





