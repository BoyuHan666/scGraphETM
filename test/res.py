import torch
import torch.nn as nn
import torchvision.models as models

# Load the pre-trained ResNet model
resnet = models.resnet50(pretrained=True)

# Modify the last fully connected layer to match the desired output size
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 512)

# Create a random input tensor with dimensions 2000x1024
input_tensor = torch.randn(2000, 1024)

# Reshape the input tensor to match the expected input shape of ResNet
input_tensor = input_tensor.view(-1, 3, 224, 224)

# Forward pass through the ResNet model
output = resnet(input_tensor)

# Reshape the output tensor to match the desired output size of 2000x512
output = output.view(2000, 512)

# Print the output tensor size
print(output.size())