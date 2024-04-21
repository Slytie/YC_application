from EMVAE import ModifiedVAETransformer, train_vae_transformer
from EMDIST4 import TransitionModel
import torch.nn as nn
import torch

# Define the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = 11
embedding_dim = 6
latent_dim = 12
num_layers = 5
heads = 2
forward_expansion = 1
dropout = 0.1
max_sequence_length = 10
epochs = 100
batch_size = 128
lr = 0.0001

# Sample execution for verification
# Sample execution for M = 10
#M = 5000
#S = 6
#N = 3
#input_array, output_array = generate_M_samples(M, S, N)
#train_loader, val_loader, test_loader = create_data_loaders(input_array, output_array)

# Sample execution for verification
dataset = TransitionModel(8000, vocab_size-1)
dataset.generate_samples()
train_loader, val_loader, test_loader = dataset.split_data()

# Initialize model and optimizer
model = ModifiedVAETransformer(vocab_size, embedding_dim, latent_dim, num_layers, heads, device, forward_expansion, dropout, max_sequence_length).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the model
train_vae_transformer(model, train_loader, val_loader, test_loader, optimizer, dataset, epochs=epochs)

print("Training completed!")