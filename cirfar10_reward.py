
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

transform = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.247, 0.243, 0.261)
)


class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class RewardModelVGG(nn.Module):
    """
    Reward model based on pretrained VGG11 for CIFAR-10.
    """
    def __init__(self, device=None):
        super(RewardModelVGG, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pretrained VGG11 model
        self.pretrained_vgg11 = self.load_vgg11_model()

        # Retain all layers, including the final classification layer
        self.features = self.pretrained_vgg11.features
        self.classifier = self.pretrained_vgg11.classifier

        # Append an additional linear layer for reward output
        self.output_layer = nn.Linear(10, 1)

        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        for param in self.output_layer.parameters():
            param.requires_grad = True


    def load_vgg11_model(self):
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True)
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Disable running statistics for BatchNorm2d layers
        #for module in model.modules():
        #    if isinstance(module, nn.BatchNorm2d):
         #       module.track_running_stats = False

        return model


    def forward(self, x):
        # Pass input through the feature extractor
        features = self.features(x)
        features = torch.flatten(features, start_dim=1)  # Flatten to [batch_size, 512]
        
        # Pass through the classifier (including the final classification layer)
        class_logits = self.classifier(features)  # Output shape: [batch_size, 10]
        
        # Pass through the appended output layer for reward prediction
        reward = self.output_layer(class_logits)  # Output shape: [batch_size, 1]
        
        return reward.squeeze(1)  # Return [batch_size]


class RewardDataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.data = np.load(dataset_path, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_i, y_i, o_i = self.data[idx]

        x_i = torch.tensor(x_i / 255.0, dtype=torch.float32)
        y_i = torch.tensor(y_i / 255.0, dtype=torch.float32)
        if self.transform:
            x_i = self.transform(x_i)
            y_i = self.transform(y_i)

        return x_i, y_i, torch.tensor(o_i, dtype=torch.float32)
        #return torch.tensor(x_i, dtype=torch.float32), torch.tensor(y_i, dtype=torch.float32), torch.tensor(o_i, dtype=torch.float32)



def reward_loss(reward_model, x_i, y_i, o_i):
    """
    Reward loss function based on the reward model and preference probabilities.
    
    Parameters
    ----------
    reward_model : RewardModel
        The reward model used to calculate reward values.
    x_i, y_i : torch.Tensor
        Feature vectors representing the ith pair of outcomes.
    o_i : float
        Preference value (0, 0.5, or 1).
    
    Returns
    -------
    loss : torch.Tensor
        Calculated loss value.
    """
    reward_x = reward_model(x_i)
    reward_y = reward_model(y_i)

    prob_y_greater_x = torch.sigmoid(reward_y - reward_x)
    prob_x_greater_y = torch.sigmoid(reward_x - reward_y)
    # prob_y_greater_x = torch.exp(reward_y) / (torch.exp(reward_x) + torch.exp(reward_y))
    
    # Compute the loss as per the formula
    loss = -((1 - o_i) * torch.log(prob_x_greater_y) + o_i * torch.log(prob_y_greater_x))
    
    return loss.mean()

def reward_loss_h(reward_model, x_i, y_i, o_i):
    """
    Reward loss function based on the reward model and preference probabilities.
    
    Parameters
    ----------
    reward_model : RewardModel
        The reward model used to calculate reward values.
    x_i, y_i : torch.Tensor
        Feature vectors representing the ith pair of outcomes.
    o_i : float
        Preference value (0, 0.5, or 1).
    
    Returns
    -------
    loss : torch.Tensor
        Calculated loss value.
    """
    if x_i.dim() == 3:
        x_i = x_i.unsqueeze(0)  # Add batch dimension if missing
    if y_i.dim() == 3:
        y_i = y_i.unsqueeze(0)
    reward_x = reward_model(x_i)
    reward_y = reward_model(y_i)

    prob_y_greater_x = torch.sigmoid(reward_y - reward_x)
    prob_x_greater_y = torch.sigmoid(reward_x - reward_y)
    # prob_y_greater_x = torch.exp(reward_y) / (torch.exp(reward_x) + torch.exp(reward_y))
    
    # Compute the loss as per the formula
    loss = -((1 - o_i) * torch.log(prob_x_greater_y) + o_i * torch.log(prob_y_greater_x))
    
    return loss.mean()

def train_reward_model(dataset_path, model_save_path, epochs=50, batch_size=64, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RewardDataset(dataset_path,transform=transform)

    # Split dataset into training and validation sets (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = RewardModelVGG().to(device)
    optimizer = optim.Adam(model.output_layer.parameters(), lr=learning_rate)  # Only update scorer

    # Initialize cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    scaler = torch.amp.GradScaler()
    # early_stopping = EarlyStopping(patience=10, save_path=model_save_path)

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for x_i, y_i, o_i in progress_bar:
            x_i, y_i, o_i = x_i.to(device), y_i.to(device), o_i.to(device).unsqueeze(1)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                train_loss = reward_loss(model, x_i, y_i, o_i)

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += train_loss.item()
            progress_bar.set_postfix({"Loss": f"{total_train_loss/ (progress_bar.n + 1):.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_i, y_i, o_i in val_loader:
                x_i, y_i, o_i = x_i.to(device), y_i.to(device), o_i.to(device).unsqueeze(1)
                with autocast():
                    val_loss = reward_loss(model, x_i, y_i, o_i)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}")

        # Step the scheduler
        scheduler.step()

    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Best model saved at {model_save_path}.")


def load_reward_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the model
    model = RewardModelVGG()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations for inference
    for param in model.features.parameters():
            param.requires_grad = False
    for param in model.classifier.parameters():
            param.requires_grad = False
    for param in model.output_layer.parameters():
        param.requires_grad = True

    return model

def load_reward_model_VGG(model_path, device=None):
    """
    Load a trained reward model and prepare it for inference.

    Parameters
    ----------
    model_path : str
        Path to the trained model file.
    device : torch.device, optional
        Device to load the model onto (e.g., 'cuda' or 'cpu').
        If not provided, defaults to GPU if available, otherwise CPU.

    Returns
    -------
    model : RewardModelVGG
        Loaded reward model, ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the model
    model = RewardModelVGG()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations for inference
    for param in model.parameters():
        param.requires_grad = False

    return model
    

def train_reward_model_nosave(dataset, epochs=50, batch_size=64, learning_rate=1e-4):
    """
    Train the reward model without saving it, and return the trained model.

    Parameters
    ----------
    dataset : list
        Dataset in (x_i, y_i, o_i) format as a list of tuples.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Learning rate for the optimizer.
    Returns
    -------
    model : RewardModel
        Trained reward model.
    """
    # Prepare the dataset
    class RewardDatasetMemory(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x_i, y_i, o_i = self.data[idx]
            return (
                x_i.clone().detach().float(),
                y_i.clone().detach().float(),
                o_i.clone().detach().float(),
            )

    dataloader = DataLoader(
        RewardDatasetMemory(dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardModelVGG().to(device)
    optimizer = torch.optim.Adam(model.output_layer.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    scaler = torch.amp.GradScaler()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        try:
            for x_i, y_i, o_i in progress_bar:
                x_i, y_i, o_i = x_i.to(device), y_i.to(device), o_i.to(device).unsqueeze(1)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda'):
                    loss = reward_loss(model, x_i, y_i, o_i)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

        except Exception as e:
            print(f"Error during training loop: {e}")
            if x_i is not None:
                print(f"x_i shape: {x_i.shape}, y_i shape: {y_i.shape}, o_i shape: {o_i.shape}")
            raise

        progress_bar.set_postfix({"Loss": f"{total_loss / (progress_bar.n + 1):.4f}"})

    return model

