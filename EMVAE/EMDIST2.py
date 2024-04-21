# Corrected full script for local execution of FullyIntegratedEMDist with necessary dependencies

import numpy as np
from itertools import product

# FullyIntegratedEMDist class definition
class FullyIntegratedEMDist:
    def __init__(self, N, M, P, S):
        self.N = N  # Number of spaces
        self.M = M  # Number of clusters per space
        self.P = P  # Number of dimensions
        self.S = S  # Number of distances (distance bins)

        self.data_spaces = []  # To store the generated synthetic data
        self.indexed_data_spaces = []  # To store the indexed synthetic data

        self.index_mapping()  # Create index mapping upon initialization

    def index_mapping(self):

        self.cluster_indices = {k: v for v, k in enumerate(product(range(self.N), range(self.M)))}
        self.dimension_indices = {k: v + self.N * self.M for v, k in enumerate(product(range(self.N), range(self.P)))}
        self.distance_indices = {k: v for v, k in enumerate(range(self.S))}

    def index_data(self):

        for space_idx, space in enumerate(self.data_spaces):
            for cluster_idx, cluster in enumerate(space):
                cluster_index = space_idx * self.M + cluster_idx
                for dimension_idx, dimension in enumerate(cluster):
                    dimension_index = self.N * self.M + space_idx * self.P + dimension_idx
                    for distance in dimension:
                        distance_index = self.distance_indices[distance]
                        indexed_entry = [
                            space_idx,
                            cluster_index,
                            dimension_index,
                            distance_index
                        ]
                        self.indexed_data_spaces.append(indexed_entry)

    def __len__(self):

        return len(self.indexed_data_spaces)

    def __getitem__(self, idx):

        return self.indexed_data_spaces[idx]

def display_data_format(data):
    """
    Display the format of the given data.
    """
    format_string = "Samples: {}, Cluster IDs: {}, Dimension IDs: {}, Distance Bins IDs: {}"
    samples = len(data)
    cluster_ids = len(data[0]) if samples > 0 else 0
    dimension_ids = len(data[0][0]) if cluster_ids > 0 else 0
    distance_bins_ids = len(data[0][0][0]) if dimension_ids > 0 else 0

    return format_string.format(samples, cluster_ids, dimension_ids, distance_bins_ids)


# Adjusting the ExtendedEMDist class according to the checklist

class FinalEMDistV3(FullyIntegratedEMDist):
    def __init__(self, N, M, P, S):
        super().__init__(N, M, P, S)
        self.transition_matrix = None
        self.output_data_spaces = []

    def generate_synthetic_data(self):
        """
        Generate synthetic data based on the provided parameters.
        Each sample should have N x M clusters, P x N unique dimension IDs, and 3 distance bins.
        """
        for _ in range(self.S):  # Adjusting for S samples
            sample = []
            cluster_ids = list(range(1, self.N * self.M + 1))  # Unique cluster IDs
            np.random.shuffle(cluster_ids)  # Randomizing the order of clusters
            for cluster_id in cluster_ids:
                cluster = [cluster_id]
                # Distributing dimension IDs across spaces
                space_id = (cluster_id - 1) // self.M
                for p in range(space_id * self.P + 1, (space_id + 1) * self.P + 1):
                    # Adjust the distance bins to be non-zero
                    distances = (np.random.choice(range(3), size=3, replace=False) + 1).tolist()
                    cluster.append(distances)
                sample.append(cluster)
            self.data_spaces.append(sample)
        # Convert the final generated data to numpy arrays
        self.data_spaces = np.array(self.data_spaces)

    def create_transition_matrix(self):
        """
        Create a sparse transition matrix to map input clusters to output clusters.
        """
        self.transition_matrix = np.zeros((self.N * self.M, self.N * self.M))

        for i in range(self.N * self.M):
            # Randomly choose an output cluster for each input cluster
            j = np.random.choice(self.N * self.M)
            self.transition_matrix[i, j] = 1

    def generate_output_data(self):
        """
        Generate output data based on the input data and the transition matrix.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix has not been defined.")

        for sample in self.data_spaces:
            output_sample = []
            for cluster in sample:
                # Use the transition matrix to find the corresponding output cluster
                output_cluster_idx = np.argmax(self.transition_matrix[cluster[0] - 1])
                output_cluster = self.data_spaces[output_cluster_idx // self.M][output_cluster_idx % self.M]
                output_sample.append(output_cluster)
            self.output_data_spaces.append(output_sample)
        # Convert the output data to numpy arrays
        self.output_data_spaces = np.array(self.output_data_spaces)


class FinalEMDistWithSplit(FinalEMDistV3):
    def __init__(self, N, M, P, S):
        super().__init__(N, M, P, S)
        self.train_input = None
        self.train_output = None
        self.val_input = None
        self.val_output = None
        self.test_input = None
        self.test_output = None

    def split_data(self, train_prop=0.8, val_prop=0.1, test_prop=0.1):
        """
        Split the data into training, validation, and test sets based on specified proportions.
        """
        # Ensure proportions sum up to 1
        assert train_prop + val_prop + test_prop == 1.0, "Proportions must sum up to 1."

        # Calculate the number of samples for each set
        total_samples = len(self.data_spaces)
        train_samples = int(total_samples * train_prop)
        val_samples = int(total_samples * val_prop)
        test_samples = total_samples - train_samples - val_samples

        # Split the input data
        self.train_input = self.data_spaces[:train_samples]
        self.val_input = self.data_spaces[train_samples:train_samples + val_samples]
        self.test_input = self.data_spaces[train_samples + val_samples:]

        # Split the output data
        self.train_output = self.output_data_spaces[:train_samples]
        self.val_output = self.output_data_spaces[train_samples:train_samples + val_samples]
        self.test_output = self.output_data_spaces[train_samples + val_samples:]


def find_highest_value(matrix):
    """
    Find the highest value in a matrix and its position.

    Args:
    - matrix (list or numpy array): The input matrix.

    Returns:
    - (value, position): A tuple containing the highest value and its position (row, column).
    """
    max_value = np.max(matrix)
    print(max_value)

Print =True

if Print ==True:

    # Testing the data split method
    test_dataset_split = FinalEMDistWithSplit(N=4, M=5, P=3, S=50)
    test_dataset_split.generate_synthetic_data()
    test_dataset_split.create_transition_matrix()
    test_dataset_split.generate_output_data()
    test_dataset_split.split_data()

    print(test_dataset_split.train_input.shape, test_dataset_split.val_input.shape, test_dataset_split.test_input.shape)
    find_highest_value(test_dataset_split.train_input)
    print(test_dataset_split.train_input)





