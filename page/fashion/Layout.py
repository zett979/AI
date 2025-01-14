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
from dash import dash_table

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

def create_classification_report_table(report_dict):
    """Convert classification report to a formatted DataFrame for Dash table"""
    # Extract metrics for each class
    classes_data = {
        key: value for key, value in report_dict.items() 
        if key not in ['accuracy', 'macro avg', 'weighted avg']
    }
    
    # Create rows for classes
    rows = []
    for class_name, metrics in classes_data.items():
        row = {
            'Class': class_name,
            'Precision': f"{metrics['precision']:.2f}",
            'Recall': f"{metrics['recall']:.2f}",
            'F1-Score': f"{metrics['f1-score']:.2f}",
            'Support': metrics['support']
        }
        rows.append(row)
    
    # Add summary rows
    rows.append({
        'Class': 'Accuracy',
        'Precision': '',
        'Recall': '',
        'F1-Score': f"{report_dict['accuracy']:.2f}",
        'Support': report_dict['macro avg']['support']
    })
    
    for avg_type in ['macro avg', 'weighted avg']:
        rows.append({
            'Class': avg_type,
            'Precision': f"{report_dict[avg_type]['precision']:.2f}",
            'Recall': f"{report_dict[avg_type]['recall']:.2f}",
            'F1-Score': f"{report_dict[avg_type]['f1-score']:.2f}",
            'Support': report_dict[avg_type]['support']
        })
    
    return rows


def Layout():
    return html.Div(
        children=[
            # File Upload Section
            html.Div(
                children=[
                    html.Div(
                        children=[
                            dcc.Upload(children="File upload", id="model-upload"),
                            html.Div(children="File name", id="model-name"),
                        ],
                        style={'marginBottom': '20px'}
                    ),
                    html.Div(
                        children=[
                            dcc.Upload(children="Dataset upload (CSV)", id="dataset-upload"),
                            html.Div(children="Dataset name", id="dataset-name"),
                        ],
                        style={'marginBottom': '20px'}
                    ),
                ],
                style={'marginBottom': '30px'}
            ),
            
            # Epsilon Slider Section
            html.Div(
                children=[
                    html.Div(children="Epsilon value for FGSM attack:"),
                    dcc.Slider(
                        id="epsilon-slider",
                        min=0.0,
                        max=0.5,
                        step=0.01,
                        value=0.0,
                        marks={i: f"{i:.2f}" for i in np.arange(0.0, 0.6, 0.1)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                style={'marginBottom': '30px'}
            ),
            
            # Results Section - Side by Side
            html.Div(
                children=[
                    # Left Side (Original Model)
                    html.Div(
                        children=[
                            html.H3(
                                "Original Model Predictions",
                                style={'textAlign': 'center', 'marginBottom': '20px'}
                            ),
                            # Confusion Matrix
                            html.Div(
                                id="confusion-matrix-before",
                                style={'textAlign': 'center', 'marginBottom': '30px'}
                            ),
                            # Classification Report
                            html.H4(
                                "Classification Report",
                                style={'textAlign': 'center', 'marginBottom': '10px'}
                            ),
                            html.Div(
                                id="classification-report-before",
                                style={'textAlign': 'center'}
                            )
                        ],
                        style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}
                    ),
                    
                    # Right Side (After Attack)
                    html.Div(
                        children=[
                            html.H3(
                                "After FGSM Attack",
                                style={'textAlign': 'center', 'marginBottom': '20px'}
                            ),
                            # Confusion Matrix
                            html.Div(
                                id="confusion-matrix-after",
                                style={'textAlign': 'center', 'marginBottom': '30px'}
                            ),
                            # Classification Report
                            html.H4(
                                "Classification Report",
                                style={'textAlign': 'center', 'marginBottom': '10px'}
                            ),
                            html.Div(
                                id="classification-report-after",
                                style={'textAlign': 'center'}
                            )
                        ],
                        style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}
                    ),
                ],
                style={
                    'display': 'flex',
                    'justifyContent': 'space-between',
                    'marginTop': '20px',
                    'width': '100%'
                }
            ),
        ],
        style={
            'maxWidth': '1600px',
            'margin': '0 auto',
            'padding': '10px'
        }
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
    [Output("dataset-name", "children"),
     Output("confusion-matrix-before", "children"),
     Output("classification-report-before", "children")],
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

        # Generate classification report
        report = classification_report(
            true_labels, 
            predictions.cpu().numpy(),
            target_names=labels_map.values(),
            output_dict=True
        )
        
        # Create table for classification report
        report_table = dash_table.DataTable(
            data=create_classification_report_table(report),
            columns=[
                {"name": "Class", "id": "Class"},
                {"name": "Precision", "id": "Precision"},
                {"name": "Recall", "id": "Recall"},
                {"name": "F1-Score", "id": "F1-Score"},
                {"name": "Support", "id": "Support"}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '100px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': -1},
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'row_index': -2},
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'row_index': -3},
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )

        return f"Processed {len(image_tensors)} images.", cm_display, report_table

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return "Error processing CSV file.", "", ""


def plot_confusion_matrix(cm):
    # Set figure size to be consistent for both matrices
    plt.figure(figsize=(8, 6))  # Adjust size for better side-by-side display
    
    # Create heatmap with consistent font sizes
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_map.values(),
        yticklabels=labels_map.values(),
        annot_kws={'size': 8}  # Adjust font size for better readability
    )
    
    # Customize labels with consistent font sizes
    plt.xlabel("Predicted", fontsize=10)
    plt.ylabel("True", fontsize=10)
    plt.title("Confusion Matrix", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)  # Increased DPI for better quality
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    
    # Return the image with specific width to ensure consistent sizing
    return html.Img(
        src=f"data:image/png;base64,{img_str}",
        style={'width': '100%', 'maxWidth': '700px'}  # Ensure consistent sizing
    )

def fgsm_attack(model, images, labels, epsilon):
    """
    Performs FGSM attack on the given images
    """
    images.requires_grad = True
    
    outputs = model(images)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    
    # Calculate gradients
    model.zero_grad()
    loss.backward()
    
    # Create perturbation
    perturbed_images = images + epsilon * images.grad.data.sign()
    
    # Ensure values stay in valid range [0,1]
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images

@callback(
    [Output("confusion-matrix-after", "children"),
     Output("classification-report-after", "children")],
    Input("epsilon-slider", "value"),
    State("dataset-upload", "contents"),
    State("model-upload", "contents"),
    prevent_initial_call=True,
)
def handle_fgsm_attack(epsilon, dataset_contents, model_contents):
    if dataset_contents is None or model_contents is None:
        return html.Div("Please upload the dataset and model before running the attack.")

    try:
        # Decode dataset contents
        content_type, content_string = dataset_contents.split(",")
        dataset_decoded = io.BytesIO(base64.b64decode(content_string))

        # Decode model contents
        model_content_type, model_content_string = model_contents.split(",")
        model_decoded = io.BytesIO(base64.b64decode(model_content_string))
        model = torch.jit.load(model_decoded)

        # Check for CUDA support
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Process CSV file
        df = pd.read_csv(dataset_decoded)
        images = df.iloc[:, 1:].values  # All columns except the first (labels)
        labels = df.iloc[:, 0].values   # The first column is the label

        # Clear and repopulate image tensors
        image_tensors.clear()
        true_labels.clear()

        # Process images
        for i in range(len(images)):
            image = images[i].reshape(28, 28).astype(np.uint8)
            image_tensors.append(transformer(Image.fromarray(image)))
            true_labels.append(labels[i])

        # Convert to tensors
        all_images_tensor = torch.stack(image_tensors).to(device)
        labels_tensor = torch.tensor(true_labels, dtype=torch.long).to(device)

        # Perform FGSM attack
        perturbed_images = fgsm_attack(
            model=model,
            images=all_images_tensor,
            labels=labels_tensor,
            epsilon=epsilon
        )

        # Get predictions on perturbed images
        with torch.no_grad():
            outputs = model(perturbed_images)
            _, perturbed_predictions = torch.max(outputs, 1)

        # Create confusion matrix
        cm = confusion_matrix(
            true_labels,
            perturbed_predictions.cpu().numpy()
        )

        # Plot confusion matrix using your existing function
        cm_display = plot_confusion_matrix(cm)
        
        # Generate classification report
        report = classification_report(
            true_labels,
            perturbed_predictions.cpu().numpy(),
            target_names=labels_map.values(),
            output_dict=True
        )
        
        # Create table for classification report
        report_table = dash_table.DataTable(
            data=create_classification_report_table(report),
            columns=[
                {"name": "Class", "id": "Class"},
                {"name": "Precision", "id": "Precision"},
                {"name": "Recall", "id": "Recall"},
                {"name": "F1-Score", "id": "F1-Score"},
                {"name": "Support", "id": "Support"}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'minWidth': '100px'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': -1},
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'row_index': -2},
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'row_index': -3},
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )

        return cm_display, report_table

    except Exception as e:
        print(f"Error during FGSM attack: {str(e)}")
        return html.Div(f"Error: {str(e)}"), ""