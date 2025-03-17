from dash import html
from page.feedback.Layout import Layout


def Feedback():
    return html.Div(
        [Layout()],
        className="relative",
    )
