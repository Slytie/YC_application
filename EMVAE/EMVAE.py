import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Hyperparameters
vocab_size = 11
embedding_dim = 300
latent_dim = 12
num_layers = 1
heads = 6
forward_expansion = 4
dropout = 0.05
max_sequence_length = 10
epochs = 10
batch_size = 128
lr = 0.0001

# Multi-head self-attention mechanism
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        )

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        inner_product = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        keys_dim_per_head = self.embed_size // self.heads
        scaled_inner_product = inner_product / (keys_dim_per_head ** (1 / 2))
        attention = torch.nn.functional.softmax(scaled_inner_product, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

# Transformer block (single block of the encoder)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Encoder
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out)

        return out

# Latent Space

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class ModifiedVAETransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_dim, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(ModifiedVAETransformer, self).__init__()

        self.encoder = Encoder(
            vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length
        )
        self.latent_space = LatentSpace(embed_size, latent_dim)
        self.decoder = ModifiedDecoder(
            vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        mu, logvar = self.latent_space(enc_out)
        z = reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z, enc_out)
        return reconstructed_x, mu, logvar


# Adjusted Loss Function
def modified_vae_loss_function(recon_x, x, mu, logvar):

    criterion = nn.CrossEntropyLoss()
    BCE = 3*criterion(recon_x.view(-1, vocab_size), x.view(-1))
    KLD = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_vae_transformer(model, train_loader, val_loader, test_loader, optimizer, dataset, epochs=5):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        for batch_idx, (input_data, output_data) in enumerate(train_loader):
            optimizer.zero_grad()
            #input_data = input_data.long().reshape(input_data.size(0), -1)
            #output_data = output_data.long().reshape(output_data.size(0), -1)
            encoded_data = model.encoder(input_data)
            mu, logvar = model.latent_space(encoded_data)
            z = reparameterize(mu, logvar)
            decoded_data, _, _ = model(input_data)
            loss = modified_vae_loss_function(decoded_data, output_data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_data, output_data in val_loader:
                input_data = input_data.long().reshape(input_data.size(0), -1)
                output_data = output_data.long().reshape(output_data.size(0), -1)
                encoded_data = model.encoder(input_data)
                mu, logvar = model.latent_space(encoded_data)
                z = reparameterize(mu, logvar)
                decoded_data, _, _ = model(input_data)
                loss = modified_vae_loss_function(decoded_data, output_data, mu, logvar)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Test Phase
    test_loss = 0
    with torch.no_grad():
        for input_data, output_data in test_loader:
            shape= output_data.shape
            print(shape)
            input_data = input_data.long().reshape(input_data.size(0), -1)
            output_data = output_data.long().reshape(output_data.size(0), -1)
            shape2= output_data.shape
            print(shape2)
            encoded_data = model.encoder(input_data)
            mu, logvar = model.latent_space(encoded_data)
            z = reparameterize(mu, logvar)
            decoded_data, _, _ = model(input_data)
            shape3= decoded_data.shape
            print(shape3)
            loss = modified_vae_loss_function(decoded_data, output_data, mu, logvar)
            cosine_similarities = dataset.compare_transition_matrices(dataset.transition_matrices_test[:10], decoded_data)
            print("Average Cosine Similarity:", np.mean(cosine_similarities))
            print(output_data[0])
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")



class LatentSpace(nn.Module):
    def __init__(self, encoder_output_dim, latent_dim):
        super(LatentSpace, self).__init__()
        self.mu_layer = nn.Linear(encoder_output_dim, latent_dim)
        self.logvar_layer = nn.Linear(encoder_output_dim, latent_dim)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

class ModifiedDecoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(ModifiedDecoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.trg_vocab_size = trg_vocab_size
        self.size_adjustment = nn.Linear(embed_size + latent_dim, embed_size )  # Adjust size from 428 to 300

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, enc_out):
        N, _ = z.shape

        # Create an empty tensor to store decoder outputs
        outputs = torch.zeros((N, max_sequence_length, self.trg_vocab_size)).to(self.device)

        # Initial input is the "start" token
        input_token = torch.zeros((N, 1), dtype=torch.long).to(self.device)

        # Iterate over each position in the target sequence
        for pos in range(max_sequence_length):
            positions = torch.arange(pos, pos + 1).expand(N, 1).to(self.device)

            # Embed the input token and add latent conditioning
            x = self.word_embedding(input_token) + self.position_embedding(positions)
            x = torch.cat([x, z.unsqueeze(1)], dim=2)  # Concatenate the latent variable to the token embeddings
            x = self.size_adjustment(x)  # Adjust the size before passing to attention

            for layer in self.layers:
                x = layer(enc_out, enc_out, x)  # Corrected the unpacking

            out_logits = self.fc_out(x)
            out = torch.nn.functional.softmax(out_logits, dim=-1)  # Apply softmax along the vocabulary dimension
            outputs[:, pos, :] = out.squeeze(1)  # Store the output

            # Use the token with the highest probability as the next input (greedy decoding)
            input_token = out.argmax(dim=2)
        return outputs
