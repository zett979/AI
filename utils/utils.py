import matplotlib.pyplot as plt
import torch
import seaborn as sns
import base64
import io
import shap
import numpy as np
from torch import nn
from dash import html


def create_classification_report_table(report_dict):
    """Convert classification report to a formatted DataFrame for Dash table"""
    # Extract metrics for each class
    classes_data = {
        key: value
        for key, value in report_dict.items()
        if key not in ["accuracy", "macro avg", "weighted avg"]
    }

    # Create rows for classes
    rows = []
    for class_name, metrics in classes_data.items():
        row = {
            "Class": class_name,
            "Precision": f"{metrics['precision']:.2f}",
            "Recall": f"{metrics['recall']:.2f}",
            "F1-Score": f"{metrics['f1-score']:.2f}",
            "Support": metrics["support"],
        }
        rows.append(row)

    # Add summary rows
    rows.append(
        {
            "Class": "Accuracy",
            "Precision": "",
            "Recall": "",
            "F1-Score": f"{report_dict['accuracy']:.2f}",
            "Support": report_dict["macro avg"]["support"],
        }
    )

    for avg_type in ["macro avg", "weighted avg"]:
        rows.append(
            {
                "Class": avg_type,
                "Precision": f"{report_dict[avg_type]['precision']:.2f}",
                "Recall": f"{report_dict[avg_type]['recall']:.2f}",
                "F1-Score": f"{report_dict[avg_type]['f1-score']:.2f}",
                "Support": report_dict[avg_type]["support"],
            }
        )

    return rows


def plot_confusion_matrix(cm, labels_map):
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
        annot_kws={"size": 8},  # Adjust font size for better readability
    )

    # Customize labels with consistent font sizes
    plt.xlabel("Predicted", fontsize=10)
    plt.ylabel("True", fontsize=10)
    plt.title("Confusion Matrix", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
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
        style={"width": "100%", "maxWidth": "700px"},  # Ensure consistent sizing
    )


def fgsm_attack(model, images, labels, epsilon):
    """
    Performs FGSM attack on the given images
    """
    # Enable gradients
    images.requires_grad_(True)

    outputs = model(images)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)

    # Calculate gradients
    model.zero_grad()
    loss.backward()

    # Create perturbation
    perturbed_images = images + epsilon * images.grad.data.sign()

    # Ensure values stay in valid range [0,1]
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    # Ensure perturbed images keep gradients
    perturbed_images.requires_grad_(True)

    return perturbed_images


def plot_shap_heatmap(shap_values, images, labels, labels_map):
    try:
        num_images = len(images)
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

        for i in range(num_images):
            predicted_class = labels[i]

            # Original Image
            orig_img = images[i].squeeze().cpu().numpy()
            axes[0, i].imshow(orig_img, cmap="gray")
            axes[0, i].set_title(f"Original: {labels_map[predicted_class]}")
            axes[0, i].axis("off")

            # SHAP Heatmap
            # Extract SHAP values for this image and remove the channel dimension
            image_shap = shap_values[i].squeeze(0)  # Remove batch dimension

            # Sum absolute values across the class dimension (last dimension)
            importance_map = np.abs(image_shap).sum(axis=-1)

            # Normalize for better visualization
            importance_map = (importance_map - importance_map.min()) / (
                importance_map.max() - importance_map.min() + 1e-8
            )

            axes[1, i].imshow(importance_map, cmap="hot")
            axes[1, i].set_title("Feature Importance")
            axes[1, i].axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return html.Img(src=f"data:image/png;base64,{img_str}", style={"width": "100%"})
    except Exception as e:
        print(f"Error in SHAP visualization: {str(e)}")
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"Images shape: {images.shape}")
        return html.Div(
            [
                html.P(f"Error visualizing SHAP values: {str(e)}"),
                html.P(f"SHAP values shape: {shap_values.shape}"),
                html.P(f"Images shape: {images.shape}"),
            ]
        )


def compute_shap_values(model, images, background):
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(images)

        # Convert to numpy array if it's not already
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
            shap_values = np.transpose(
                shap_values, (1, 2, 3, 4, 0)
            )  # Rearrange to (N, C, H, W, num_classes)

        print(f"SHAP values type: {type(shap_values)}")
        print(f"Shape of SHAP values: {shap_values.shape}")
        return shap_values

    except Exception as e:
        print(f"Error in SHAP computation: {e}")
        raise


def plot_shap_heatmap_after(shap_values, images, labels, labels_map):
    try:
        num_images = len(images)
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

        # Convert labels to numpy if it's a tensor
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        for i in range(num_images):
            # Handle label conversion
            predicted_class = int(labels[i])

            # Handle image tensor conversion
            if isinstance(images, torch.Tensor):
                orig_img = images[i].detach().cpu().numpy()
            else:
                orig_img = images[i]

            # Original Image
            orig_img = orig_img.squeeze()  # Remove channel dimension if present
            axes[0, i].imshow(orig_img, cmap="gray")
            axes[0, i].set_title(f"Original: {labels_map[predicted_class]}")
            axes[0, i].axis("off")

            # SHAP Heatmap
            # Handle different SHAP value formats
            if isinstance(shap_values, torch.Tensor):
                image_shap = shap_values[i].detach().cpu().numpy()
            else:
                image_shap = shap_values[i]

            image_shap = image_shap.squeeze(0)  # Remove batch dimension

            # Sum absolute values across the class dimension
            importance_map = np.abs(image_shap).sum(axis=-1)

            # Normalize with small epsilon to prevent division by zero
            importance_map = (importance_map - importance_map.min()) / (
                importance_map.max() - importance_map.min() + 1e-8
            )

            axes[1, i].imshow(importance_map, cmap="hot")
            axes[1, i].set_title("Feature Importance")
            axes[1, i].axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return html.Img(src=f"data:image/png;base64,{img_str}", style={"width": "100%"})
    except Exception as e:
        print(f"Error in SHAP visualization: {str(e)}")
        return html.Div(
            [
                html.P(f"Error visualizing SHAP values: {str(e)}"),
                html.P(
                    f"SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'unknown'}"
                ),
                html.P(
                    f"Images shape: {images.shape if hasattr(images, 'shape') else 'unknown'}"
                ),
            ]
        )


def compute_shap_values_after(model, images, background):
    try:
        model.eval()

        if background.requires_grad:
            background = background.detach()

        explainer = shap.GradientExplainer(model, background)

        if images.requires_grad:
            images_detached = images.detach()
        else:
            images_detached = images

        shap_values = explainer.shap_values(images_detached)

        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
            shap_values = np.transpose(shap_values, (1, 2, 3, 4, 0))

        print(f"SHAP values type: {type(shap_values)}")
        print(f"Shape of SHAP values: {shap_values.shape}")
        return shap_values

    except Exception as e:
        print(f"Error in SHAP computation: {e}")
        raise
