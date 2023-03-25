import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(
        self,
        lr,
        input_dim,
        output_dim,
    ):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Network layers
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, self.output_dim)

        # Loss function and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else
                                   'cpu')
        self.to(self.device)

    def forward(self, state):
        # Forward pass
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions

    def save_checkpoint(self, path):
        # Save model
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        # Load model
        self.load_state_dict(torch.load(path))
        self.eval()


if __name__ == "__main__":
    raise SystemExit("This is not a script.")
