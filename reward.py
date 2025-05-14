import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

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

    prob_y_greater_x = torch.exp(reward_y) / (torch.exp(reward_x) + torch.exp(reward_y))
    
    # Compute the loss as per the formula
    loss = -((1 - o_i) * torch.log(1 - prob_y_greater_x) + o_i * torch.log(prob_y_greater_x))
    
    return loss.mean()

def train_reward_model(reward_dataset_path, model_save_path, epochs=10, lr=1e-4):
    reward_data = np.load(reward_dataset_path, allow_pickle=True)
    x_i_sample = reward_data[0][0]
    input_dim = x_i_sample.shape[0]
    
    reward_model = RewardModel(input_dim)
    optimizer = optim.Adam(reward_model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x_i, y_i, o_i in reward_data:
            x_i = torch.tensor(x_i, dtype=torch.float32)
            y_i = torch.tensor(y_i, dtype=torch.float32)
            o_i = torch.tensor(o_i, dtype=torch.float32)

            x_i = (x_i - x_i.mean()) / x_i.std()
            y_i = (y_i - y_i.mean()) / y_i.std()

            optimizer.zero_grad()
            loss = reward_loss(reward_model, x_i, y_i, o_i)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(reward_data)}")
    
    torch.save(reward_model.state_dict(), model_save_path)
    print(f"Reward model trained and saved at {model_save_path}")

def load_reward_model(model_path, input_dim):
    """
    Load the pre-trained reward model from a file.
    
    Parameters
    ----------
    model_path : str
        Path to the saved reward model file.
    input_dim : int
        Dimension of the input feature vector.

    Returns
    -------
    reward_model : RewardModel
        Loaded reward model.
    """
    reward_model = RewardModel(input_dim)
    reward_model.load_state_dict(torch.load(model_path))
    reward_model.eval()  # Set to evaluation mode
    return reward_model

def train_reward_model_nosave(dataset, epochs=10, lr=1e-4):
    reward_data = dataset
    x_i_sample = reward_data[0][0]
    input_dim = x_i_sample.shape[0]
    
    reward_model = RewardModel(input_dim)
    optimizer = optim.Adam(reward_model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x_i, y_i, o_i in reward_data:
            x_i = torch.tensor(x_i, dtype=torch.float32)
            y_i = torch.tensor(y_i, dtype=torch.float32)
            o_i = torch.tensor(o_i, dtype=torch.float32)

            x_i = (x_i - x_i.mean()) / x_i.std()
            y_i = (y_i - y_i.mean()) / y_i.std()

            optimizer.zero_grad()
            loss = reward_loss(reward_model, x_i, y_i, o_i)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(reward_data)}")
    
    return reward_model