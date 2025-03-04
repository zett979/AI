import torchvision.transforms as transform
from dash import html, dcc, Output, Input, State, callback
from components.Button import Button
from components.Typography import P


def TensorAndModelConfig():
    return html.Div(
        children=[
            html.Div(
                children=[
                    P("Input Tensor Dimensions:", variant="body1", className="mb-2"),
                    html.Div(
                        className="flex gap-2 items-center",
                        children=[
                            html.Div(
                                className="flex gap-1 items-center",
                                children=[
                                    html.Label("Batch:", className="text-sm"),
                                    dcc.Input(
                                        id="dim-batch",
                                        type="number",
                                        value=1,
                                        min=1,
                                        className="input w-16",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="flex gap-1 items-center",
                                children=[
                                    html.Label("Channels:", className="text-sm"),
                                    dcc.Input(
                                        id="dim-channels",
                                        type="number",
                                        value=1,
                                        min=1,
                                        className="input w-16",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="flex gap-1 items-center",
                                children=[
                                    html.Label("Height:", className="text-sm"),
                                    dcc.Input(
                                        id="dim-height",
                                        type="number",
                                        value=28,
                                        min=1,
                                        className="input w-16",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="flex gap-1 items-center",
                                children=[
                                    html.Label("Width:", className="text-sm"),
                                    dcc.Input(
                                        id="dim-width",
                                        type="number",
                                        value=28,
                                        min=1,
                                        className="input w-16",
                                    ),
                                ],
                            ),
                            Button(
                                "Apply",
                                id="apply-dimensions",
                                variant="secondary",
                                size="sm",
                            ),
                        ],
                    ),
                ],
                className="px-2 py-3 rounded-md",
            ),
            html.Div(
                children=[
                    Button(
                        [
                            dcc.Upload(children="Model upload", id="model-upload"),
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
                    Button(
                        children=[
                            html.Img(
                                src="/assets/images/icons/edit.svg",
                                className="size-5",
                            ),
                        ],
                        variant="primary",
                        id="labels-modify",
                        size="sm",
                        n_clicks=0,
                    ),
                ],
                className="flex gap-3",
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
    )


# Input tensor shape callback
@callback(
    Output("dim-height", "value"),
    Output("dim-width", "value"),
    Input("apply-dimensions", "n_clicks"),
    State("dim-batch", "value"),
    State("dim-channels", "value"),
    State("dim-height", "value"),
    State("dim-width", "value"),
    prevent_initial_call=True,
)
def update_tensor_dimensions(n_clicks, batch, channels, height, width):
    global input_tensor_shape, transformer
    input_tensor_shape = [batch, channels, height, width]

    # Update the transformer to match the new dimensions
    transformer = transform.Compose(
        [
            transform.Grayscale(channels),
            transform.Resize((height, width)),
            transform.ToTensor(),
        ]
    )

    print(f"Updated input tensor shape to: {input_tensor_shape}")

    return height, width
