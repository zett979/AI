import base64
import torch
import io
import pandas as pd
from PIL import Image
from dash import html, callback, dcc, State, Output, Input
import torchvision.transforms as transform
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive)
import matplotlib.pyplot as plt
import numpy as np

# Initialize global variables
image_tensors = []
true_labels = []

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

transformer = transform.Compose(
    [transform.Grayscale(1), transform.Resize((28, 28)), transform.ToTensor()]
)


def Layout():
    return html.Div(
        children=[
            dcc.Upload(children="File upload", id="model-upload"),
            html.Div(children="File name", id="model-name"),
            dcc.Upload(children="Dataset upload (CSV)", id="dataset-upload"),
            html.Div(children="Dataset name", id="dataset-name"),
            html.Div(id="confusion-matrix"),
        ]
    )


@callback(
    Output("model-name", "children"),
    Input("model-upload", "contents"),
    State("model-upload", "filename"),
    prevent_initial_call=True,
)
def handleFileUpload(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = io.BytesIO(base64.b64decode(content_string))
    model: torch.jit.ScriptModule = torch.jit.load(decoded)

    # Check for CUDA support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the right device

    # Print the model's output shape with a dummy input tensor
    input_tensor = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        tensor: torch.Tensor = model(input_tensor)
        print(tensor.shape[1])

    return model.original_name


@callback(
    Output("dataset-name", "children"),  # Output for the dataset status
    Output("confusion-matrix", "children"),  # Output for the confusion matrix image
    Input("dataset-upload", "contents"),
    State("model-upload", "contents"),
    prevent_initial_call=True,
)
def handle_csv_upload(contents, model_contents):
    if contents is None:
        return "Please upload a CSV file.", ""

    if model_contents is None:
        return "Please upload a model first.", ""

    # Decode the uploaded CSV file
    content_type, content_string = contents.split(",")
    decoded = io.BytesIO(base64.b64decode(content_string))

    # Decode the uploaded model
    model_content_type, model_content_string = model_contents.split(",")
    model_decoded = io.BytesIO(base64.b64decode(model_content_string))
    model = torch.jit.load(model_decoded)

    # Check for CUDA support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the right device

    try:
        # Process the CSV file containing images and labels
        df = pd.read_csv(decoded)

        # Split the data into images and labels
        images = df.iloc[:, 1:].values  # All columns except the first (labels)
        labels = df.iloc[:, 0].values   # The first column is the label

        # Normalize and transform images
        image_tensors.clear()
        true_labels.clear()

        for i in range(len(images)):
            image = images[i].reshape(28, 28).astype(np.uint8)  # Reshape to 28x28
            label = labels[i]
            image_tensors.append(transformer(Image.fromarray(image)))  # Apply transformation
            true_labels.append(label)

        all_images_tensor = torch.stack(image_tensors)  # Combine into a tensor

        # Get predictions
        model.eval()
        all_images_tensor = all_images_tensor.to(device)  # Move images to the correct device
        print("Model evaluation")
        with torch.no_grad():
            outputs = model(all_images_tensor)
            _, predictions = torch.max(outputs, 1)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions.cpu().numpy())
        cm_display = plot_confusion_matrix(cm)

        return f"Processed {len(image_tensors)} images.", cm_display

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return "Error processing CSV file.", ""


def plot_confusion_matrix(cm):
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_map.values(),
        yticklabels=labels_map.values(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save to a buffer and encode as base64 to display in Dash
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Return the image in a format Dash can display
    return html.Img(src=f"data:image/png;base64,{img_str}")
