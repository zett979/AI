import dash
from dash import dcc, html, Input, Output, State, no_update
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import plotly.express as px
import base64
from io import BytesIO
from extras.fashion_model import CNN

model = CNN()
model.load_state_dict(torch.load('./extras/fashion_mnist_cnn.pth'))  
model.eval()

app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("Fashion MNIST Image Upload and FGSM Attack"),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload Image'),
        multiple=False
    ),
    
    html.Div(id='image-container', children=[]),
    
    # Epsilon value slider for FGSM attack
    html.Div([
        html.Label("Epsilon Value:"),
        dcc.Slider(
            id='epsilon-slider',
            min=0.01,
            max=1.0,
            step=0.01,
            value=0.1,
            marks={i / 10: f'{i / 10:.1f}' for i in range(1, 11)},
        ),
    ], style={'width': '60%', 'padding': '20px'}),
    
    # Button to apply FGSM attack
    html.Button('Apply FGSM Attack', id='fgsm-button', n_clicks=0),
    
    # Display original and adversarial images and predictions
    html.Div(id='comparison-container', children=[]),
])

def decode_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(BytesIO(decoded))
    img = img.convert('L')  # Convert to grayscale (Fashion MNIST)
    img = img.resize((28, 28))  # Resize to 28x28 (Fashion MNIST size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for the model
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Helper function to apply FGSM attack
def fgsm_attack(model, image, epsilon=0.1):
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    
    # Target class is the class with the highest prediction
    loss = nn.CrossEntropyLoss()(output, output.max(1)[1])
    
    # Backward pass to calculate gradients
    model.zero_grad()
    loss.backward()
    
    # Collect the gradients of the image
    image_grad = image.grad.data
    
    # Apply FGSM
    perturbed_image = image + epsilon * image_grad.sign()
    
    # Clipping to ensure the image remains valid (i.e., pixel values between 0 and 1)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

# Helper function to get predictions
def get_prediction(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

def tensor_to_image(tensor):
    # Detach the tensor from the computation graph to avoid gradient issues
    tensor = tensor.detach()  # Detach the tensor to remove it from the computation graph
    
    # Remove the batch dimension (which is 1 in this case) and the channel dimension
    tensor = tensor.squeeze(0)  # Now shape should be (28, 28)
    tensor = tensor.squeeze(0)  # Remove channel dimension, should now be shape (28, 28)
    
    tensor = tensor.cpu().numpy()  # Convert to numpy array
    tensor = tensor * 0.5 + 0.5  # Denormalize to [0, 1]
    
    return tensor

# Callback to handle image upload and FGSM attack
@app.callback(
    Output('image-container', 'children'),
    Output('comparison-container', 'children'),
    Input('upload-image', 'contents'),
    Input('fgsm-button', 'n_clicks'),
    Input('epsilon-slider', 'value'),
    State('upload-image', 'filename')
)
def update_output(uploaded_image, n_clicks, epsilon, filename):
    if uploaded_image is None:
        raise dash.no_update

    # Decode the uploaded image
    img_tensor = decode_image(uploaded_image)
    
    # Get the prediction for the original image
    original_pred = get_prediction(model, img_tensor)
    
    # Create a Plotly figure to display the original image
    original_img = tensor_to_image(img_tensor)
    
    # If FGSM button is clicked, apply the attack
    if n_clicks > 0:
        perturbed_image = fgsm_attack(model, img_tensor, epsilon=epsilon)
        perturbed_pred = get_prediction(model, perturbed_image)
        
        # Convert the perturbed image tensor to an image for display
        perturbed_img = tensor_to_image(perturbed_image)
        
        # Create Plotly figures to compare the original and perturbed images
        original_fig = px.imshow(original_img, title=f"Original Image\nPredicted Class: {original_pred}")
        perturbed_fig = px.imshow(perturbed_img, title=f"Adversarial Image\nPredicted Class: {perturbed_pred}")
        
        return [
            html.Div([
                html.H3(f"Uploaded Image: {filename}"),
                dcc.Graph(figure=original_fig)
            ]),
            html.Div([
                dcc.Graph(figure=perturbed_fig)
            ])
        ]
    
    # If no attack is applied, just show the original image and its prediction
    original_fig = px.imshow(original_img, title=f"Original Image\nPredicted Class: {original_pred}")
    
    return [
        html.Div([
            html.H3(f"Uploaded Image: {filename}"),
            dcc.Graph(figure=original_fig)
        ]),
        html.Div([])
    ]

if __name__ == '__main__':
    app.run_server(debug=True)
