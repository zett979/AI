from components.Typography import P
from api.ollama import getModelName, chatToModel
from dash import (
    html,
    callback,
    Input,
    Output,
    dcc,
)
import dash

PROMPT = ""
LOADING = False
CLICKS = 0


def Layout():
    return html.Div(
        children=[
            dcc.Store(
                data=([]),
                storage_type="session",
                id="chats",
            ),
            html.Div(
                children=[
                    P("OLLAMA", variant="heading1", id="clear-btn", n_clicks=0),
                    html.Div(
                        [
                            P(
                                "Currently using",
                                variant="body2",
                                className="text-[#9BADAD]",
                            ),
                            P(getModelName(), variant="body2", id="model-name"),
                        ],
                        className="flex gap-1",
                    ),
                ],
                className="flex flex-col gap-2",
            ),
            html.Div(
                children=[
                    html.Div(id="chat-output", className="flex flex-col gap-2"),
                ],
                className="w-full h-[70vh] relative overflow-y-auto flex flex-col gap-3 p-5 bg-[#E3F4F4]",
            ),
            html.Div(
                children=[
                    dcc.Input(
                        placeholder="Ask anything...",
                        className="w-full bg-[#D2E9E9] p-3",
                        id="chat-input",
                        value="",
                        type="text",
                        disabled=False,
                        autoComplete="off",
                    ),
                    html.Button(
                        html.Img(
                            src="/assets/images/icons/send.svg",
                            className="size-7",
                        ),
                        id="chat-btn",
                        n_clicks_timestamp=0,
                        className="size-10 absolute right-5 top-1 flex justify-center items-center bg-[#D2E9E9]",
                    ),
                    html.Button(
                        id="real-chat-btn", className="hidden", n_clicks_timestamp=0
                    ),
                ],
                className="w-full sticky top-[67vh] rounded-lg overflow-hidden",
            ),
        ],
        className="relative flex flex-col gap-3 px-10 my-2",
    )


@callback(Output("chat-input", "value"), Input("chat-input", "value"))
def chatInput(input):
    return input


@callback(
    Output("chats", "data", allow_duplicate=True),
    Output("real-chat-btn", "n_clicks_timestamp"),
    Output("chat-input", "disabled", allow_duplicate=True),
    Input("chats", "data"),
    Input("chat-btn", "n_clicks_timestamp"),
    Input("chat-input", "value"),
    prevent_initial_call=True,
)
def onChatClick(chats, n_clicks, input):
    global LOADING, CLICKS
    if n_clicks > CLICKS and LOADING == False and input != "":
        new_chats = chats.copy()
        new_chats.append({"type": "User", "message": input})  # User message
        new_chats.append({"type": "Loading", "message": ""})  # Loading placeholder
        CLICKS = n_clicks
        LOADING = True
        return new_chats, n_clicks, True
    else:
        return dash.no_update


@callback(
    Output("chats", "data", allow_duplicate=True),
    Output("chat-input", "value", allow_duplicate=True),
    Output("chat-input", "disabled", allow_duplicate=True),
    Input("chats", "data"),
    Input("real-chat-btn", "n_clicks_timestamp"),
    Input("chat-input", "value"),
    prevent_initial_call=True,
    background=True,
)
def onChat(chats, n_clicks, input):
    global LOADING
    new_chats = chats.copy()
    if LOADING == True:
        response = chatToModel(input)  # Get response from Ollama
        new_chats.pop()  # Remove the loading message
        new_chats.append({"type": "AI", "message": response})  # Add AI response
        LOADING = False
        return new_chats, "", False
    else:
        return dash.no_update


@callback(Output("chat-output", "children"), Input("chats", "data"))
def updateChats(chats):
    children = []
    for chat in chats:
        children.append(
            html.Div(
                children=f"{chat['message']}",
                className=f"flex w-fit max-w-[300px] p-2 bg-[#C4DFDF] rounded-md {'ml-auto' if chat['type'] == 'User' else ''} {'animate-pulse !w-32 h-5' if chat['type'] == 'Loading' else ''}",
            )
        )
    return children


@callback(Output("chats", "data"), Input("clear-btn", "n_clicks"))
def clearChats(n_clicks):
    if n_clicks > 0:
        return []
    else:
        return dash.no_update
