import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transforms import  complex_cellular_automata_transform as transform
from transforms import plot_predictions

# Function to create synthetic 2D data
def create_synthetic_2d_data(num_samples, image_size, transform_function):
    data = np.random.rand(num_samples, image_size, image_size)
    targets = np.array([transform_function(x) for x in data])
    return data, targets

def transform_function(image):
    return image * 0.5 + 0.2 * image ** 2

# Downward block (encoding part)
class DownwardBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownwardBlock2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pooled = self.pool(x_conv)
        return x_pooled, x_conv

# Upward block (decoding part)
class UpwardBlock2D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpwardBlock2D, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x_up = self.up(x)
        x_concat = torch.cat((skip, x_up), dim=1)
        x_conv = self.conv(x_concat)
        return x_conv

# Transformer Attention Block for 2D Data
class TransformerAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(TransformerAttentionBlock2D, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.embedding_size = in_channels // num_heads
        self.query = nn.Linear(self.embedding_size, self.embedding_size)
        self.key = nn.Linear(self.embedding_size, self.embedding_size)
        self.value = nn.Linear(self.embedding_size, self.embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels, self.in_channels),
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape
        x_reshaped = x.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
        x_heads = x_reshaped.reshape(-1, self.num_heads, self.embedding_size)
        queries = self.query(x_heads)
        keys = self.key(x_heads)
        values = self.value(x_heads)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.embedding_size ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.reshape(batch_size, height * width, self.in_channels)
        output = self.ffn(attended_values)
        output_reshaped = output.permute(0, 2, 1).view(batch_size, self.in_channels, height, width)
        return output_reshaped

# Complete U-Net with transformer attention
class CompleteTransformerUNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(CompleteTransformerUNet2D, self).__init__()
        self.down1 = DownwardBlock2D(in_channels, 16)
        self.down2 = DownwardBlock2D(16, 32)
        self.down3 = DownwardBlock2D(32, 64)
        self.attention = TransformerAttentionBlock2D(64, num_heads)
        self.up3 = UpwardBlock2D(64, 32, 32)
        self.up2 = UpwardBlock2D(32, 16, 16)
        self.up1 = UpwardBlock2D(16, 16, 16)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1_pooled, x1_conv = self.down1(x)
        x2_pooled, x2_conv = self.down2(x1_pooled)
        x3_pooled, x3_conv = self.down3(x2_pooled)
        attended_output = self.attention(x3_pooled)
        x3_upward = self.up3(attended_output, x3_conv)
        x2_upward = self.up2(x3_upward, x2_conv)
        x1_upward = self.up1(x2_upward, x1_conv)
        x_out = self.final_conv(x1_upward)
        return x_out


# Modifying the create_synthetic_2d_data function to create binary 2D data (0 or 1)
def create_synthetic_2d_data(num_samples, image_size, transform_function, fraction_of_ones=0.50):
    data = np.random.choice([0, 1], size=(num_samples, image_size, image_size), p=[1 - fraction_of_ones, fraction_of_ones])
    targets = np.array([transform_function(x) for x in data])
    return data, targets


# Create synthetic data
new_image_size = 32
num_train_samples = 10000
num_test_samples = 200
new_train_data_2d, new_train_targets_2d = create_synthetic_2d_data(num_train_samples, new_image_size, transform)
new_test_data_2d, new_test_targets_2d = create_synthetic_2d_data(num_test_samples, new_image_size, transform)
new_train_data_2d_tensor = torch.tensor(new_train_data_2d, dtype=torch.float32).unsqueeze(1)
new_train_targets_2d_tensor = torch.tensor(new_train_targets_2d, dtype=torch.float32).unsqueeze(1)
new_test_data_2d_tensor = torch.tensor(new_test_data_2d, dtype=torch.float32).unsqueeze(1)
new_test_targets_2d_tensor = torch.tensor(new_test_targets_2d, dtype=torch.float32).unsqueeze(1)
new_batch_size = 8
new_train_dataset_2d = TensorDataset(new_train_data_2d_tensor, new_train_targets_2d_tensor)
new_test_dataset_2d = TensorDataset(new_test_data_2d_tensor, new_test_targets_2d_tensor)
new_train_loader_2d = DataLoader(new_train_dataset_2d, batch_size=new_batch_size, shuffle=True)
new_test_loader_2d = DataLoader(new_test_dataset_2d, batch_size=new_batch_size, shuffle=False)


# Initialize the model, loss function, and optimizer
complete_transformer_unet_model_2d = CompleteTransformerUNet2D(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(complete_transformer_unet_model_2d.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    complete_transformer_unet_model_2d.train()
    train_loss = 0.0
    for batch_data, batch_targets in new_train_loader_2d:
        optimizer.zero_grad()
        outputs = complete_transformer_unet_model_2d(batch_data)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_data.size(0)
    train_loss /= len(new_train_loader_2d.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss}")

# Testing
complete_transformer_unet_model_2d.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_data, batch_targets in new_test_loader_2d:
        outputs = complete_transformer_unet_model_2d(batch_data)
        loss = criterion(outputs, batch_targets)
        test_loss += loss.item() * batch_data.size(0)
test_loss /= len(new_test_loader_2d.dataset)
print(f"Test Loss: {test_loss}")

# Number of random samples to plot
N = 5

# Call the plot_predictions function to plot the results
plot_predictions(complete_transformer_unet_model_2d, new_test_data_2d_tensor, new_test_targets_2d_tensor, N)
