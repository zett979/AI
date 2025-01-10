from dash import html
from page.fashion.Layout import Layout


def Fashion():
    return html.Div(
        [Layout()],
        className="relative",
    )
