import matplotlib.pyplot as plt
import torch
import seaborn as sns
import base64
import io

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
