import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.config import Config
from src.detection import CustomDataset, get_model

def train_model():
    # Load configuration settings
    config = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    
    train_dataset = CustomDataset(config.train_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model
    model = get_model(config.model_name).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print epoch loss
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Save the trained model
    model_save_path = os.path.join(config.model_save_path, 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    train_model()