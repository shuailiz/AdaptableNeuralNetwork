import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN architecture
class SimpleCNNAdaptable(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.weight_matrix = np.random.random((200, 200))
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32*32*32, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
    def forward(self, x, weights_mapping):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        with torch.no_grad():
            self.fc1.weight.copy_(self.weight_matrix[weights_mapping[0]])
            self.fc2.weight.copy_(self.weight_matrix[weights_mapping[1]])

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

    def update_weight_matrix(self, weights_mapping):
        self.weight_matrix[weights_mapping[0]] = self.fc1.weight.clone()..detach().numpy()
        self.weight_matrix[weights_mapping[1]] = self.fc2.weight.clone()..detach().numpy()