import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters
dim = 100  # Define a 100x100 grid for simplicity
K = 100    # Proportionality constant for number of steps

# Initialize walker state at the center of the grid
walker_state = torch.zeros((4, dim, dim), dtype=torch.complex128)
walker_state[:, dim//2, dim//2] = 1/2  # Start with equal superposition of all directions

# Define the Grover coin
C_G = 0.5 * torch.tensor([[-1, 1, 1, 1],
                          [1, -1, 1, 1],
                          [1, 1, -1, 1],
                          [1, 1, 1, -1]], dtype=torch.complex128)

# Define the external field as a simple gradient
def phi(x, y):
    return 0.01 * (x + y)  # Linear gradient for demonstration

# Modify the coin operation based on the external field
def coin_operation_with_field(walker_state):
    phase_matrix = torch.tensor([[torch.exp(1j * phi(x, y)) for x in range(dim)] for y in range(dim)], dtype=torch.complex128)
    phase_matrix_4x4 = torch.diag_embed(torch.stack([phase_matrix] * 4))
    modified_coin = torch.matmul(C_G, phase_matrix_4x4)
    return torch.einsum('ij,jkl->ikl', modified_coin, walker_state)

# Define the shift operation
def shift_operation(walker_state):
    up_state = torch.roll(walker_state[0], shifts=-1, dims=1)
    down_state = torch.roll(walker_state[1], shifts=1, dims=1)
    left_state = torch.roll(walker_state[2], shifts=-1, dims=2)
    right_state = torch.roll(walker_state[3], shifts=1, dims=2)
    return torch.stack([up_state, down_state, left_state, right_state])

# Random number of steps based on the field at the starting position
num_steps = int(K * abs(phi(dim//2, dim//2)))

# Quantum walk simulation
for _ in range(num_steps):
    walker_state = coin_operation_with_field(walker_state)
    walker_state = shift_operation(walker_state)

# Final state distribution (probability)
prob_distribution = torch.sum(torch.abs(walker_state)**2, dim=0)

# Plotting the probability distribution
plt.figure(figsize=(10, 8))
plt.imshow(prob_distribution.numpy(), cmap='hot', interpolation='nearest', origin='lower')
plt.colorbar(label="Probability")
plt.title("Probability Distribution after 2D Quantum Walk with External Field")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()
