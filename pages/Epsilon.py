import dash
from page.Epsilon import Epsilon

dash.register_page(__name__, path="/epsilon")

layout = Epsilon()
