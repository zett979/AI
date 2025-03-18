import dash
from dash import Dash, html, Output, Input, ctx, callback
from components.Button import Button
from page.DataDialog import DataDialog
from utils.CacheManager import background_callback_manager

app = Dash(
    __name__, use_pages=True, background_callback_manager=background_callback_manager
)
IS_OPEN = False
app.layout = html.Div(
    [
        DataDialog(),
        html.Div(
            [
                Button(
                    children="Home",
                    variant="primary_ghost",
                    size="sm",
                    asLink=True,
                    href="/",
                ),
                Button(
                    children="Chat with",
                    variant="primary_ghost",
                    size="sm",
                    asLink=True,
                    href="/chat",
                ),
                html.Div(
                    children=[
                        Button(
                            children=[
                                "Analysis",
                                html.Img(src="/assets/images/icons/arrow-down.svg", id="analysis-arrow", className="duration-150"),
                            ],
                            variant="primary_ghost",
                            size="sm",
                            n_clicks=0,
                            id="analysis",
                            className="flex items-center gap-2.5",
                        ),
                        html.Div(
                            children=[
                                Button(
                                    children="Model Analysis",
                                    variant="primary_ghost",
                                    size="sm",
                                    asLink=True,
                                    href="/epsilon",
                                    className="w-[150px]"
                                ),
                                Button(
                                    children="Dataset Analysis",
                                    variant="primary_ghost",
                                    size="sm",
                                    asLink=True,
                                    href="/analysis",
                                    className="w-[150px]"
                                ),
                            ],
                            id="analysis-dropdown",
                            className=f"flex flex-col gap-2.5 absolute top-full px-2 py-3 bg-[#EEFFFF] backdrop-blur-lg right-0 w-fit shadow-md z-[10] rounded-lg duration-150 origin-top group-hover:scale-100 scale-0 after:absolute after:absolute after:top-0 after:left-0 after:z-[-1] after:w-full after:h-[200px] after:scale-200 after:bg-transparent after:pointer-none",
                        ),
                    ],
                    className="relative group",
                ),
                Button(
                    children="Feedbacks",
                    variant="primary_ghost",
                    size="sm",
                    asLink=True,
                    href="/feedback",
                ),
                # Button(
                #     children=["Login", html.Img(src="/assets/images/icons/login.svg")],
                #     variant="primary",
                #     size="sm",
                #     asLink=True,
                #     href="/login",
                #     className="flex items-center gap-2.5",
                # ),
                # Button(
                #     children="Signup",
                #     variant="primary_ghost",
                #     size="sm",
                #     asLink=True,
                #     href="/signup",
                # ),
            ],
            style={"boxShadow": "0 0 25px 0 hsla(180, 30%, 82%, 0.50)"},
            className="w-full sticky top-0 left-0 flex gap-5 justify-end items-center px-20 py-3 backdrop-blur-sm bg-[#EEFFFF]/80 z-[100]",
        ),
        dash.page_container,
    ],
    className="relative",
)

if __name__ == "__main__":
    app.run(debug=True)
