import base64
import torch
import io
import pandas as pd
import torchvision.transforms as transform
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # Use Agg backend (non-interactive)
import matplotlib.pyplot as plt
import numpy as np
import dash

from components.Button import Button
from components.Typography import P
from PIL import Image
from dash import (
    html,
    callback,
    dcc,
    State,
    Output,
    Input,
    ALL,
)
from sklearn.metrics import classification_report, confusion_matrix
from dash import dash_table
from utils.utils import (
    create_classification_report_table,
    plot_confusion_matrix,
    fgsm_attack,
    compute_shap_values,
    plot_shap_heatmap,
    plot_shap_heatmap_after,
)

# Initialize global variables
image_tensors = []
true_labels = []

labels_map = {0: ""}

transformer = transform.Compose(
    [transform.Grayscale(1), transform.Resize((28, 28)), transform.ToTensor()]
)


def Layout():
    return html.Div(
        children=[
            # File Upload Section
            html.Div(
                children=[
                    html.Div(
                        children=[
                            Button(
                                [
                                    dcc.Upload(
                                        children="File upload", id="model-upload"
                                    ),
                                    html.Div(
                                        children="File name",
                                        id="model-name",
                                        style={"display": "none"},
                                    ),
                                ],
                                variant="primary",
                                size="sm",
                                className="flex gap-2",
                                id="model-upload-button",
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            Button(
                                [
                                    dcc.Upload(
                                        children="Dataset upload (CSV)",
                                        id="dataset-upload",
                                    ),
                                ],
                                variant="primary",
                                size="sm",
                                id="dataset-upload-button",
                            )
                        ],
                    ),
                ],
                className="flex flex-col gap-3",
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
                style={"marginBottom": "30px"},
            ),
            # Results Section - Side by Side
            html.Div(
                children=(
                    [
                        # Left Side (Original Model)
                        html.Div(
                            children=[
                                P(
                                    "Original Model Predictions",
                                    variant="body1",
                                    className="text-center",
                                ),
                                html.Div(
                                    className="w-full h-72 bg-[#C4DFDF] animate-pulse",
                                    style={"display": "none"},
                                    id="loading-predictions-before",
                                ),
                                # Confusion Matrix
                                html.Div(
                                    id="confusion-matrix-before",
                                    className="text-center",
                                ),
                                # Classification Report
                                P(
                                    "Classification Report",
                                    variant="body1",
                                    className="text-center",
                                ),
                                html.Div(
                                    className="w-full h-72 bg-[#C4DFDF] animate-pulse",
                                    style={"display": "none"},
                                    id="loading-report-before",
                                ),
                                html.Div(
                                    id="classification-report-before",
                                    style={"textAlign": "center"},
                                ),
                                P(
                                    "Shap Report",
                                    variant="body1",
                                    className="text-center",
                                ),
                                html.Div(
                                    className="w-full h-72 bg-[#C4DFDF] animate-pulse",
                                    style={"display": "none"},
                                    id="loading-shap-before",
                                ),
                                html.Div(
                                    id="shap-visualization-before",
                                ),
                            ],
                            className="w-full flex flex-col gap-3",
                        ),
                        # Right Side (After Attack)
                        html.Div(
                            children=[
                                P(
                                    "After FGSM Attack",
                                    variant="body1",
                                    className="text-center",
                                ),
                                html.Div(
                                    className="w-full h-72 bg-[#C4DFDF] animate-pulse",
                                    style={"display": "none"},
                                    id="loading-predictions-after",
                                ),
                                # Confusion Matrix
                                html.Div(
                                    id="confusion-matrix-after", className="text-center"
                                ),
                                # Classification Report
                                P(
                                    "Classification Report",
                                    variant="body1",
                                    className="text-center",
                                ),
                                html.Div(
                                    className="w-full h-72 bg-[#C4DFDF] animate-pulse",
                                    style={"display": "none"},
                                    id="loading-report-after",
                                ),
                                html.Div(
                                    id="classification-report-after",
                                    className="text-center",
                                ),
                                P(
                                    "Shap Report",
                                    variant="body1",
                                    className="text-center",
                                ),
                                html.Div(
                                    className="w-full h-72 bg-[#C4DFDF] animate-pulse",
                                    style={"display": "none"},
                                    id="loading-shap-after",
                                ),
                                html.Div(
                                    id="shap-visualization-after",
                                ),
                            ],
                            className="w-full flex flex-col gap-3",
                        ),
                    ]
                ),
                className="grid xl:grid-cols-2 grid-cols-1 gap-5",
            ),
            # Label Dialog
            html.Div(
                html.Div(
                    children=[
                        html.Button(
                            id="label-close-btn",
                            children=[
                                html.Img(
                                    src="/assets/images/icons/cross.svg",
                                    className="size-6",
                                )
                            ],
                            className="absolute right-3 top-3",
                        ),
                        html.Div(
                            children=[
                                P("Label Setup", variant="body1"),
                                # New Button for Uploading Labels CSV
                                html.Div(
                                    id="labels-upload-container",
                                    children=[
                                        Button(
                                            [
                                                dcc.Upload(
                                                    children="Upload Labels CSV",
                                                    id="labels-upload",
                                                ),
                                            ],
                                            variant="primary",
                                            size="sm",
                                            id="labels-upload-button",
                                            style={
                                                "display": "none"
                                            },  # Initially hidden
                                        ),
                                    ],
                                ),
                            ],
                            className="flex gap-2 items-center",
                        ),
                        html.Div(
                            children=[],
                            id="label-inputs",
                            className="flex flex-col gap-2",
                        ),
                    ],
                    className="w-[814px] h-[400px] overflow-auto flex flex-col gap-2 fixed left-[50%] top-[50%] -translate-x-[50%] -translate-y-[50%] p-5 rounded-xl bg-[#D2E9E9] z-[120] duration-300",
                    style={"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)"},
                ),
                id="label-dialog",
                className="w-full h-full hidden fixed left-0 top-0 z-[120] bg-black/20",
                style={"display": "none"},
            ),
        ],
        className="relative px-10 my-4 flex flex-col gap-4",
    )


@callback(
    Output("labels-upload-button", "style"),
    Input("model-upload", "contents"),
)
def show_labels_upload_button(model_contents):
    if model_contents is not None:
        return {"display": "block"}  # Show the button
    return {"display": "none"}  # Hide the button


@callback(
    Output("label-inputs", "children", allow_duplicate=True),
    Input("labels-upload", "contents"),
    prevent_initial_call=True,
)
def handle_labels_upload(label_contents):
    if label_contents is None:
        return dash.no_update

    # Decode the uploaded CSV file
    label_content_type, label_content_string = label_contents.split(",")
    label_decoded = io.BytesIO(base64.b64decode(label_content_string))

    try:
        df = pd.read_csv(label_decoded)
        print(df)
        labels = df.iloc[:, 0].values

        labels_list = list(labels)[:10]  # Ensure max 10 labels
        while len(labels_list) < 10:
            labels_list.append("")  # Fill missing labels with ""

        # Update the labels_map
        global labels_map
        labels_map = {i: label for i, label in enumerate(labels_list)}

        # Generate the input components dynamically based on labels_map
        label_inputs = [
            html.Div(
                children=[
                    html.Label(f"Label {i}"),
                    dcc.Input(
                        id={"type": "label-input", "index": i},
                        type="text",
                        value=label,
                        className="input flex-1",
                    ),
                ],
                className="flex flex-col gap-2",
            )
            for i, label in labels_map.items()
        ]

        return label_inputs

    except Exception as e:
        print(f"Error processing labels CSV file: {e}")
        return dash.no_update


@callback(
    Output("label-inputs", "children"),
    Input({"type": "label-input", "index": ALL}, "value"),
)
def update_labels(input_values):
    print("Input values:", input_values)
    global labels_map

    # Check if any input value has changed and update the labels_map accordingly
    for i, value in enumerate(input_values):
        if value != labels_map.get(i, ""):  # Only update if the value has changed
            labels_map[i] = value
            print(f"Updated Label {i}: {value}")

    # Generate the input components dynamically based on labels_map
    label_inputs = [
        html.Div(
            children=[
                html.Label(f"Label {i}"),
                dcc.Input(
                    id={"type": "label-input", "index": i},
                    type="text",
                    value=labels_map.get(i, ""),
                    className="input flex-1",
                ),
            ],
            className="flex flex-col gap-2",
        )
        for i in range(
            len(labels_map)
        )  # Ensure we are generating inputs for all available labels
    ]

    return label_inputs


@callback(
    Output("label-dialog", "style", allow_duplicate=True),
    Input("label-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_dialog(n_clicks):
    if n_clicks > 0:
        return {"display": "none"}
    else:
        dash.no_update


@callback(
    Output("model-name", "children"),
    Output("model-name", "style"),
    Output("label-dialog", "style"),
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
    for module in model.modules():
        if hasattr(module, "in_features"):
            sample_shape = (1, module.in_features)
            print("In features", sample_shape)
            break
        elif hasattr(module, "input_dim"):
            sample_shape = (1, module.input_dim)
            print("In features", sample_shape)
            break
    # Print the model's output shape with a dummy input tensor
    input_tensor = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        tensor: torch.Tensor = model(input_tensor)
        i = 0
        while i < tensor.shape[1]:
            labels_map[i] = ""
            i += 1
    return (
        " - " + model.original_name,
        {"display": "block"},
        {"display": "block"},
    )


@callback(
    [
        Output("confusion-matrix-before", "children"),
        Output("classification-report-before", "children"),
        Output("loading-predictions-before", "style", allow_duplicate=True),
        Output("loading-report-before", "style", allow_duplicate=True),
        Output("loading-shap-before", "style", allow_duplicate=True),
        Output("dataset-upload-button", "disabled", allow_duplicate=True),
        Output("model-upload-button", "disabled", allow_duplicate=True),
        Output("shap-visualization-before", "children"),
    ],
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
        labels = df.iloc[:, 0].values  # The first column is the label

        # Normalize and transform images
        image_tensors.clear()
        true_labels.clear()

        for i in range(len(images)):
            image = images[i].reshape(28, 28).astype(np.uint8)  # Reshape to 28x28
            label = labels[i]
            image_tensors.append(
                transformer(Image.fromarray(image))
            )  # Apply transformation
            true_labels.append(label)

        all_images_tensor = torch.stack(image_tensors)  # Combine into a tensor

        # Get predictions
        model.eval()
        all_images_tensor = all_images_tensor.to(
            device
        )  # Move images to the correct device
        print("Model evaluation")
        with torch.no_grad():
            outputs = model(all_images_tensor)
            _, predictions = torch.max(outputs, 1)

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions.cpu().numpy())
        cm_display = plot_confusion_matrix(cm, labels_map)

        # Generate classification report
        report = classification_report(
            true_labels,
            predictions.cpu().numpy(),
            target_names=labels_map.values(),
            output_dict=True,
        )

        # Create table for classification report
        report_table = dash_table.DataTable(
            data=create_classification_report_table(report),
            columns=[
                {"name": "Class", "id": "Class"},
                {"name": "Precision", "id": "Precision"},
                {"name": "Recall", "id": "Recall"},
                {"name": "F1-Score", "id": "F1-Score"},
                {"name": "Support", "id": "Support"},
            ],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "10px", "minWidth": "100px"},
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": -1},
                    "fontWeight": "bold",
                    "backgroundColor": "rgb(248, 248, 248)",
                },
                {
                    "if": {"row_index": -2},
                    "fontWeight": "bold",
                    "backgroundColor": "rgb(248, 248, 248)",
                },
                {
                    "if": {"row_index": -3},
                    "fontWeight": "bold",
                    "backgroundColor": "rgb(248, 248, 248)",
                },
            ],
        )

        shapey_labels = df.iloc[:, 0].values

        # Shapey values
        try:

            shapey_images = df.iloc[:, 1:].values.reshape(-1, 28, 28)

            processed_images = []
            for img in shapey_images[:100]:
                tensor = transformer(Image.fromarray(img.astype(np.uint8)))
                processed_images.append(tensor)

            background = torch.stack(processed_images).to(device)

            # Process sample images
            sample_processed = []
            for img in shapey_images[:5]:  # Use first 5 images as samples
                tensor = transformer(Image.fromarray(img.astype(np.uint8)))
                sample_processed.append(tensor)

            sample_images = torch.stack(sample_processed).to(device)

            # Compute SHAP values
            shap_values = compute_shap_values(model, sample_images, background)
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            shap_plot = html.Div("Error computing SHAP values")
        return (
            cm_display,
            report_table,
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            False,
            False,
            plot_shap_heatmap(
                shap_values, sample_images, shapey_labels[:5], labels_map
            ),
        )

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return "Error processing CSV file.", "", ""


@callback(
    [
        Output("confusion-matrix-after", "children"),
        Output("classification-report-after", "children"),
        Output("loading-predictions-after", "style", allow_duplicate=True),
        Output("loading-report-after", "style", allow_duplicate=True),
        Output("loading-shap-after", "style", allow_duplicate=True),
        Output("dataset-upload-button", "disabled", allow_duplicate=True),
        Output("model-upload-button", "disabled", allow_duplicate=True),
        Output("shap-visualization-after", "children"),
    ],
    Input("epsilon-slider", "value"),
    State("dataset-upload", "contents"),
    State("model-upload", "contents"),
    prevent_initial_call=True,
)
def handle_fgsm_attack(epsilon, dataset_contents, model_contents):
    if dataset_contents is None or model_contents is None:
        return html.Div(
            "Please upload the dataset and model before running the attack."
        )

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
        labels = df.iloc[:, 0].values  # The first column is the label

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
            model=model, images=all_images_tensor, labels=labels_tensor, epsilon=epsilon
        )

        # Get predictions on perturbed images
        with torch.no_grad():
            outputs = model(perturbed_images)
            _, perturbed_predictions = torch.max(outputs, 1)

        # Create confusion matrix
        cm = confusion_matrix(true_labels, perturbed_predictions.cpu().numpy())

        # Plot confusion matrix using your existing function
        cm_display = plot_confusion_matrix(cm, labels_map)

        # Generate classification report
        report = classification_report(
            true_labels,
            perturbed_predictions.cpu().numpy(),
            target_names=labels_map.values(),
            output_dict=True,
        )

        # Create table for classification report
        report_table = dash_table.DataTable(
            data=create_classification_report_table(report),
            columns=[
                {"name": "Class", "id": "Class"},
                {"name": "Precision", "id": "Precision"},
                {"name": "Recall", "id": "Recall"},
                {"name": "F1-Score", "id": "F1-Score"},
                {"name": "Support", "id": "Support"},
            ],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "10px", "minWidth": "100px"},
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": -1},
                    "fontWeight": "bold",
                    "backgroundColor": "rgb(248, 248, 248)",
                },
                {
                    "if": {"row_index": -2},
                    "fontWeight": "bold",
                    "backgroundColor": "rgb(248, 248, 248)",
                },
                {
                    "if": {"row_index": -3},
                    "fontWeight": "bold",
                    "backgroundColor": "rgb(248, 248, 248)",
                },
            ],
        )

        shapey_labels = df.iloc[:, 0].values
        try:
            perturbed_cpu = perturbed_images.cpu()

            background = perturbed_cpu[:100]

            sample_images = perturbed_cpu[:5]

            shapey_labels = df.iloc[:, 0].values[:5]

            shap_values = compute_shap_values(model, sample_images, background)

            shap_plot = plot_shap_heatmap(
                shap_values, sample_images, shapey_labels, labels_map
            )

        except Exception as e:
            print(f"Error computing SHAP values for perturbed images: {e}")
            shap_plot = html.Div("Error computing SHAP values for perturbed images")

        return (
            cm_display,
            report_table,
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            False,
            False,
            plot_shap_heatmap(
                shap_values, sample_images, shapey_labels[:5], labels_map
            ),
        )

    except Exception as e:
        print(f"Error during FGSM attack: {str(e)}")
        return (
            html.Div(f"Error: {str(e)}"),
            "",
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            False,
            False,
            html.Div(""),
        )


@callback(
    Output("loading-predictions-before", "style"),
    Output("loading-report-before", "style"),
    Output("loading-shap-before", "style"),
    Output("dataset-upload-button", "disabled"),
    Output("model-upload-button", "disabled"),
    Input("dataset-upload", "contents"),
    State("model-upload", "contents"),
    prevent_initial_call=True,
)
def onBeforeCalculation(contents, model):
    return {"display": "block"}, {"display": "block"}, {"display": "block"}, True, True


@callback(
    Output("loading-predictions-after", "style"),
    Output("loading-report-after", "style"),
    Output("loading-shap-after", "style"),
    Output("confusion-matrix-after", "children", allow_duplicate=True),
    Output("classification-report-after", "children", allow_duplicate=True),
    Input("epsilon-slider", "value"),
    prevent_initial_call=True,
)
def onAfterCalculation(epsilon):
    return {"display": "block"}, {"display": "block"}, {"display": "block"}, [], []
