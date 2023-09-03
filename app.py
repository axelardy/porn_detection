import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

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

single_image_transform = transforms.Compose([
    transforms.Resize(200),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']

        if image_file and allowed_file(image_file.filename):
            image = Image.open(image_file)
            image = single_image_transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                predicted = torch.round(output).item()

            if predicted == 1:
                prediction = 'Negative'
            else:
                prediction = 'Positive'
            print(prediction)
            return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, port=5000)