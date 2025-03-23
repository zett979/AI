from components.Typography import P
from api.ollama import getModelName, chatToModel
from dash import html, callback, Input, Output, dcc, State
import dash

PROMPT = ""
CLICKS = 0


def Layout():
    return html.Div(
        children=[
            dcc.Store(
                data=({"chats": [], "isLoading": False}),
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
                        className="w-full bg-[#E3F4F4] placeholder:text-[#9BADAD] border border-transparent focus:border-[#9BADAD] p-3 duration-150",
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
                className="w-full sticky top-[67vh] rounded-lg",
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
    Input("chat-btn", "n_clicks_timestamp"),
    State("chats", "data"),
    State("chat-input", "value"),
    prevent_initial_call=True,
)
def onChatClick(n_clicks, chat_data, input):
    chats = chat_data["chats"]
    LOADING = chat_data["isLoading"]
    global CLICKS
    if n_clicks is None or (n_clicks <= CLICKS and n_clicks != 0):
        return dash.no_update

    if not LOADING and input.strip():
        if chats is None:
            chats = []

        new_chats = chats + [
            {"type": "User", "message": input},
            {"type": "Loading", "message": ""},
        ]

        CLICKS = n_clicks
        print("onChatClick - Updated chats:", new_chats)

        return {"chats": new_chats, "isLoading": True}, n_clicks, True

    return dash.no_update


@callback(
    Output("chats", "data", allow_duplicate=True),
    Output("chat-input", "value", allow_duplicate=True),
    Output("chat-input", "disabled", allow_duplicate=True),
    Input("real-chat-btn", "n_clicks_timestamp"),
    State("chats", "data"),
    State("chat-input", "value"),
    prevent_initial_call=True,
    background=True,
)
def onChat(n_clicks, chat_data, input):
    chats = chat_data["chats"]
    LOADING = chat_data["isLoading"]
    if n_clicks is None:
        return dash.no_update

    if LOADING:
        if chats is None:
            chats = []  

        new_chats = chats.copy()

        response = chatToModel(input)  

        if new_chats and new_chats[-1]["type"] == "Loading":
            new_chats.pop()

        new_chats.append({"type": "AI", "message": response}) 

        return {"chats": new_chats, "isLoading": False}, "", False

    return dash.no_update


@callback(Output("chat-output", "children"), Input("chats", "data"))
def updateChats(chat_data):
    chats = chat_data["chats"]
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
