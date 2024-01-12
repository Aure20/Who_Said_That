import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import tensorflow_hub as hub
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        # Forward pass through the encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.softmax(decoded)
    


class CustomDataset(Dataset):
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)

        # Load the Universal Sentence Encoder
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get text from the CSV file
        text1 = self.data.iloc[idx, 0]
        text2 = self.data.iloc[idx, 1]

        # Encode the text using Universal Sentence Encoder
        embeddings1 = self.encode_sentence(text1)
        embeddings2 = self.encode_sentence(text2)

        return torch.tensor(embeddings1, dtype=torch.float32), torch.tensor(embeddings2, dtype=torch.float32)

    def encode_sentence(self, sentence):
        # Use Universal Sentence Encoder to encode the sentence
        embeddings = self.embed([sentence]).numpy()
        return embeddings.squeeze()


def train(model, train_loader, validation_loader, criterion, optimizer, num_epochs, patience=4):
    best_loss = float('inf')
    patience_counter = 0

    val_losses = []
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Training Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.8f}')

        # Validation
        with torch.no_grad():
            model.eval()
            for inputs, labels in validation_loader:
                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels) 

                running_loss += loss.item()

            val_loss = running_loss / len(validation_loader)
            val_losses.append(val_loss)
            print(f'Training Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.8f}')

            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping. Patience reached.")
                break
    return train_losses, val_losses
# Example usage:
csv_file_path = "train.csv"
custom_dataset = CustomDataset(csv_file_path)

# Split the dataset into training and validation sets
train_dataset, validation_dataset = random_split(custom_dataset, [0.8,0.2])

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# Create the autoencoder model
model = Autoencoder()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with early stopping
num_epochs = 200

"""
train_losses, val_losses = train(model, train_loader, validation_loader, criterion, optimizer, num_epochs)

torch.save(model.state_dict(), 'autoencoder_model.pth')

import matplotlib.pyplot as plt
# Plotting
plt.plot(val_losses, label='Validation Loss', color='blue')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')  
plt.legend() 
plt.show()
plt.savefig('loss.png')
"""