from dash import html, dcc, Input, Output, ctx, callback
from components.Button import Button


def Layout():
    return html.Div(
        [
            html.Div(
                children=[
                    html.P("Feedback", className="text-4xl font-bold text-center"),
                ]
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.P(
                                "Your name",
                                className="text-xl font-semibold",
                            ),
                            dcc.Input(
                                id="name",
                                type="text",
                                placeholder="Please enter your name...",
                                className="w-full h-16 bg-[#E3F4F4] placeholder:text-[#9BADAD] border border-transparent focus:border-[#9BADAD] text-xl rounded-md px-2 py-3",
                            ),
                        ],
                        className="flex flex-col gap-4",
                    ),
                    html.Div(
                        children=[
                            html.P(
                                "What is your issue?",
                                className="text-xl font-semibold",
                            ),
                            dcc.Dropdown(
                                id="feedback",
                                searchable=True,
                                options=[
                                    "Data analysis",
                                    "Model Analysis",
                                    "Shape Analysis",
                                    "Chat application",
                                    "Others",
                                ],
                                placeholder="Please select your issue...",
                            ),
                        ],
                        className="flex flex-col gap-4",
                    ),
                    html.Div(
                        children=[
                            html.P(
                                "Describe the issues in detail...",
                                className="text-xl font-semibold",
                            ),
                            dcc.Textarea(
                                id="details",
                                rows=5,
                                draggable=False,
                                placeholder="Please describe the issue in detail...",
                                className="w-full h-40 bg-[#E3F4F4] placeholder:text-[#9BADAD] border border-transparent focus:border-[#9BADAD] text-xl rounded-md px-2 py-3",
                            ),
                        ],
                        className="flex flex-col gap-4",
                    ),
                    html.Div(
                        children=[
                            Button(
                                "Submit",
                                id="submit-feedback",
                                variant="primary",
                                className="max-w-[180px] text-2xl",
                            ),
                            Button(
                                "Cancel",
                                id="cancel-feedback",
                                variant="primary_ghost",
                                className="max-w-[180px] text-2xl",
                            ),
                        ],
                        className="flex flex gap-4 justify-center items-center w-fit mx-auto",
                    ),
                ],
                className="flex flex-col gap-5 justify-center w-full mx-auto max-w-[1028px] mx-[11.5%]",
            ),
        ],
        className="relative py-16 flex flex-col gap-6 items-center",
    )
