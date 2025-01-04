from dash import html
from page.chat.Layout import Layout


def Chat():
    return html.Div(
        [Layout()],
        className="relative",
    )
