import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import requests
import zipfile
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import base64

# Define the Fast Gradient Sign Method (FGSM) attack
def fgsm_attack(model, loss_fn, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * data.grad.sign()
    perturbed_data = data + perturbation
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Ensure pixel values are valid
    return perturbed_data

# Define the MNIST CNN model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #12, 12, 10
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) #4, 4, 20
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize Dash App
app = dash.Dash(__name__)

# Layout of the Dash Application
app.layout = html.Div([
    html.H1("FGSM Detection in MNIST Dataset", style={'text-align': 'center'}),
    
    html.Div([
        dcc.Upload(
            id='upload-model',
            children=html.Button('Upload Pre-trained Model (.pth file)'),
            multiple=False
        ),
        html.Div(id='output-model-upload', style={'margin-top': '20px'}),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    
    html.Div([
        dcc.Upload(
            id='upload-dataset',
            children=html.Button('Upload MNIST Dataset (ZIP)'),
            multiple=False
        ),
        html.Div(id='output-dataset-upload', style={'margin-top': '20px'}),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    
    html.Div([
        html.Label("Adjust Epsilon for FGSM Attack:"),
        dcc.Slider(
            id='epsilon-slider',
            min=0,
            max=0.5,
            step=0.01,
            value=0.1,
            marks={i/10: f'{i/10}' for i in range(6)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Div(id='slider-output', style={'text-align': 'center', 'margin-top': '20px'})
    ], style={'margin-top': '20px'}),
    
    html.Div([
        html.H3("Model Performance:"),
        html.P(id="accuracy-output", style={'text-align': 'center'}),
        
        html.H3("Visualizing Clean vs Adversarial Examples:"),
        dcc.Graph(id='image-plot')
    ], style={'text-align': 'center', 'margin-top': '20px'})
])

### Step 3: Create Callbacks to Handle Uploads and Model Evaluation

# Global variables to store the model and dataset
model = None
dataset_loader = None

# Define callback for model upload
@app.callback(
    Output('output-model-upload', 'children'),
    Output('output-dataset-upload', 'children'),
    Input('upload-model', 'contents'),
    Input('upload-dataset', 'contents'),
    State('upload-model', 'filename'),
    State('upload-dataset', 'filename'),
)
def upload_files(model_content, dataset_content, model_filename, dataset_filename):
    global model, dataset_loader
    
    # Check if the model is uploaded
    if model_content is not None:
        content_type, content_string = model_content.split(',')
        decoded = base64.b64decode(content_string)
        model = CNN()
        model.load_state_dict(torch.load(io.BytesIO(decoded), map_location=torch.device('cpu')))
        model.eval()
        model_message = f"Model {model_filename} loaded successfully."
    else:
        model_message = "Upload a trained model (.pth file)."

    # Check if dataset is uploaded
    if dataset_content is not None:
        content_type, content_string = dataset_content.split(',')
        decoded = base64.b64decode(content_string)
        with zipfile.ZipFile(io.BytesIO(decoded)) as zf:
            zf.extractall("/tmp/dataset")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.ImageFolder(root="/tmp/dataset", transform=transform)
        dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        dataset_message = f"Dataset {dataset_filename} loaded successfully."
    else:
        dataset_message = "Upload a dataset (ZIP file containing MNIST-like images)."
    
    return model_message, dataset_message


# Define callback for epsilon slider to evaluate and visualize
@app.callback(
    Output('slider-output', 'children'),
    Output('accuracy-output', 'children'),
    Output('image-plot', 'figure'),
    Input('epsilon-slider', 'value'),
)
def evaluate_model(epsilon):
    global model, dataset_loader

    if model is None or dataset_loader is None:
        return "Please upload both the model and dataset first.", "", {}

    loss_fn = nn.CrossEntropyLoss()

    # Evaluate clean and adversarial accuracy
    correct_clean = 0
    correct_adv = 0
    total = 0
    images = []
    labels = []
    adv_images = []

    for data, target in dataset_loader:
        data, target = data.cuda(), target.cuda()
        
        # Clean data accuracy
        output_clean = model(data)
        _, predicted_clean = torch.max(output_clean, 1)
        correct_clean += (predicted_clean == target).sum().item()
        
        # Generate adversarial examples
        data_adv = fgsm_attack(model, loss_fn, data, target, epsilon)
        
        # Adversarial data accuracy
        output_adv = model(data_adv)
        _, predicted_adv = torch.max(output_adv, 1)
        correct_adv += (predicted_adv == target).sum().item()
        
        total += target.size(0)
        
        # Store images to display
        images.append(data[0].cpu().detach().numpy().squeeze())
        adv_images.append(data_adv[0].cpu().detach().numpy().squeeze())
        labels.append(target[0].item())

    clean_accuracy = correct_clean / total * 100
    adv_accuracy = correct_adv / total * 100

    # Displaying accuracy output
    accuracy_output = f"Clean Accuracy: {clean_accuracy:.2f}% | Adversarial Accuracy (ε={epsilon}): {adv_accuracy:.2f}%"

    # Visualize the first image (clean vs adversarial)
    fig = go.Figure()
    fig.add_trace(go.Image(z=images[0], name='Clean Image'))
    fig.add_trace(go.Image(z=adv_images[0], name='Adversarial Image'))

    return f"epsilon = {epsilon}", accuracy_output, fig

if __name__ == '__main__':
    app.run_server(debug=True)
