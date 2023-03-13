import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define hyperparameters
input_dim = 20
output_dim = 20
hidden_dim = 128
num_layers = 2
num_heads = 4
dropout_prob = 0.1
batch_size = 32
num_epochs = 100
learning_rate = 1e-3

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout_prob):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout_prob
            ),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(input_dim * batch_size, output_dim)
    
    def forward(self, x):
        x = x.reshape(batch_size, -1, 20)
        # x is of shape (batch_size, 10, 20)
        x = x.permute(1, 0, 2) # swap batch and sequence dimension
        x = self.encoder(x) # apply the Transformer encoder
        x = x.reshape(x.shape[0], -1) # flatten the output
        x = self.decoder(x) # apply a linear layer to get the predicted output
        x = x.reshape(-1, 20) # reshape the output to be of shape (batch_size, 20)
        return x

# Generate some sample data
num_groups = 1000
data = np.random.randint(1, 81, size=(num_groups, 20))
labels = np.roll(data, -10, axis=0)

# Split the data into train and test sets
train_data = data[:800]
train_labels = labels[:800]
test_data = data[800:]
test_labels = labels[800:]

# Convert the data and labels to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Create a PyTorch dataset and data loader for the training data
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the Transformer model and the optimizer
model = TransformerModel(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout_prob)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the loss function
loss_fn = nn.MSELoss()

# Train the model
for epoch in range(num_epochs):
    for i, (batch_data, batch_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch_data)
        loss = loss_fn(output, batch_labels)
        loss.backward()
        optimizer.step()
    print("Epoch {}, Loss: {:.4f}".format(epoch+1, loss.item()))

# Evaluate the model on the test set
with torch.no_grad():
    test_output = model(test_data)
    test_loss = loss_fn(test_output, test_labels)
    print("Test Loss: {:.4f}".format(test_loss.item()))
