from dash import html
from page.epsilon.epsilon import Layout


def Epsilon():
    return html.Div(
        [Layout()],
        className="relative",
    )
