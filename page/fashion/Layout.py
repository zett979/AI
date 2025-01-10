import base64
import torch
import zipfile
import io
from PIL import Image
from dash import html, callback, dcc, State, Output, Input
import torch.utils.data.dataloader
import torchvision.transforms as transform

transformer = transform.Compose([
    transform.Grayscale(1),
    transform.Resize((28, 28)),
    transform.ToTensor()
])

def Layout():
    return html.Div(
        children=[
            dcc.Upload(children="File upload", id="model-upload"),
            html.Div(children="File name", id="model-name"),
            dcc.Upload(children="Dataset upload", id="dataset-upload"),
            html.Div(children="Dataset name", id="dataset-name")
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
    last_layer = list(model.named_children())[-1][0]
    input_tensor = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        tensor: torch.Tensor = model(input_tensor)
        print(tensor.shape[1])
    return model.original_name

def get_features(model, layer_name, input_tensor):
    features = {}

    def hook_fn(module, input, output):
        features[layer_name] = output.detach()

    # Register hook on the layer you're interested in
    hook = getattr(model, layer_name).register_forward_hook(hook_fn)

    # Run the input through the model
    with torch.no_grad():
        model(input_tensor)

    # Remove the hook after use
    hook.remove()

    return features[layer_name]

@callback(
    Output("dataset-name", "children"),
    Input("dataset-upload", "contents"),
    prevent_initial_call=True,
)
def handle_zip_upload(contents):
    if contents is None:
        return "Please upload a ZIP file."

    # Decode the uploaded ZIP file
    content_type, content_string = contents.split(",")
    decoded = io.BytesIO(base64.b64decode(content_string))

    # Extract the ZIP file and transform images
    image_tensors = []
    with zipfile.ZipFile(decoded) as z:
        for file_info in z.infolist():
            # Only process valid image files
            if file_info.filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.avif')):
                try:
                    with z.open(file_info.filename) as img_file:
                        image_file = Image.open(img_file).convert("RGB")
                        transformed_image = transformer(image_file)
                        image_tensors.append(transformed_image.unsqueeze(0))  
                except (IOError, OSError, Image.UnidentifiedImageError) as e:
                    print(f"Skipping file {file_info.filename}: {e}")

    if image_tensors:
        all_images_tensor = torch.cat(image_tensors)  
        print(all_images_tensor.shape)  # Print the shape of the combined tensor

    return "Processed images: {}".format(len(image_tensors))
