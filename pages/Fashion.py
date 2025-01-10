import dash
from page.Fashion import Fashion

dash.register_page(__name__, path="/fashion")

layout = Fashion()
