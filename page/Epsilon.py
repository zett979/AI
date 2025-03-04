from dash import html
from page.epsilon.Layout import Layout


def Epsilon():
    return html.Div(
        [Layout()],
        className="relative",
    )
