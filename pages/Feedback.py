import dash
from page.Feedback import Feedback

dash.register_page(__name__, path="/feedback")

layout = Feedback()
