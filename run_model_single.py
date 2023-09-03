import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load the pre-trained model
model = Net()
checkpoint = torch.load('model/model_checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the transformation for the single image
single_image_transform = transforms.Compose([
    transforms.Resize(200),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load a single image
image_path = 'test_negative.jpg'  # Replace with the path to your single image
image = Image.open(image_path)
image = single_image_transform(image).unsqueeze(0)  # Add a batch dimension

# Use the model for prediction
with torch.no_grad():
    output = model(image)
    predicted = torch.round(output)

# Print the prediction (1 for positive class, 0 for negative class)
if predicted.item() == 1:
    print("Predicted: Negative Class")
else:
    print("Predicted: Positive Class")
