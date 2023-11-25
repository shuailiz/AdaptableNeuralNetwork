import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN architecture
class SimpleCNNAdaptation(nn.Module):
    def __init__(self, num_input_dim):
        super(SimpleCNNAdaptation, self).__init__()
        
        # Fully connected layers
        # Number of output features (128) + number of classes (2)
        self.fc1 = nn.Linear(in_features=num_input_dim, out_features=300*300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=300*300, out_features=3000*3000)
        
    def forward(self, x, prediction):
        x = torch.cat((x, prediction), 1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        
        return x
