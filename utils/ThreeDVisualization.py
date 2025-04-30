import numpy as np
import torch
import io
import base64
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import traceback


def plot_shap_3d_scatter(
    shap_values, input_images, true_labels, labels_map
):
    """
    Creates a combined 3D visualization of SHAP values from multiple images
    with different colors for each sample's positive and negative SHAP values.
    """
    try:
        if isinstance(input_images, torch.Tensor):
            input_images = input_images.cpu().numpy()

        abs_shap_values = np.abs(shap_values)
        num_samples = min(3, len(input_images))

        color_schemes = [
            {"positive": "red", "negative": "darkred"},
            {"positive": "green", "negative": "darkgreen"},
            {"positive": "blue", "negative": "darkblue"},
        ]

        fig = go.Figure()

        for i in range(num_samples):
            img_shap = shap_values[i].squeeze()
            abs_img_shap = abs_shap_values[i].squeeze()

            if len(img_shap.shape) > 2:
                img_shap = np.mean(img_shap, axis=0)
                abs_img_shap = np.mean(abs_img_shap, axis=0)

            height, width = img_shap.shape

            x, y = np.meshgrid(np.arange(width), np.arange(height))

            max_abs_val = np.max(abs_img_shap)
            threshold = max_abs_val * 0.1  # Show top 90% significant values

            pos_mask = img_shap > threshold
            neg_mask = img_shap < -threshold

            # Add positive SHAP points (with unique color for each sample)
            fig.add_trace(
                go.Scatter3d(
                    x=y[pos_mask].flatten(),
                    y=x[pos_mask].flatten(),
                    z=abs_img_shap[pos_mask].flatten(),
                    mode="markers",
                    marker=dict(size=8, color=color_schemes[i]["positive"], opacity=0.8),
                    name=f"Positive SHAP ({ labels_map[true_labels[i]] if labels_map[true_labels[i]] else 'Sample Image'} {i+1})",
                )
            )

            # Add negative SHAP points (with unique color for each sample)
            fig.add_trace(
                go.Scatter3d(
                    x=y[neg_mask].flatten(),
                    y=x[neg_mask].flatten(),
                    z=-abs_img_shap[neg_mask].flatten(),
                    mode="markers",
                    marker=dict(size=8, color=color_schemes[i]["negative"], opacity=0.8),
                    name=f"Negative SHAP ({ labels_map[true_labels[i]] if labels_map[true_labels[i]] else 'Sample Image'} {i+1})",
                )
            )

        fig.update_layout(
            title="Combined 3D SHAP Visualization",
            scene=dict(
                xaxis_title="Width",
                yaxis_title="Height",
                zaxis_title="SHAP Value",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            height=700,
        )

        return html.Div(dcc.Graph(figure=fig))
        
    except Exception as e:
        error_msg = f"Error generating 3D SHAP visualization: {str(e)}\n"
        error_msg += traceback.format_exc()
        print(error_msg)
        return html.Div(
            html.Pre(error_msg, style={"whiteSpace": "pre-wrap", "color": "red"})
        )
