import dash
from page.Chat import Chat

dash.register_page(__name__, path="/chat")

layout = Chat()
