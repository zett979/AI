import streamlit as st
from streamlit_elements import elements, mui, html
from components.Base import ButtonStyle, BaseStyle

BaseStyle()


def NavBar():
    with elements("style_elements_css"):
        html.div(
            children=[
                mui.Button("Login", sx=ButtonStyle()),
                mui.Button("Register", sx=ButtonStyle(ghost=True, size="sm")),
            ],
            style={
                "width": "100%",
                "display": "flex",
                "justifyContent": "end",
                "gap": "30px",
                "padding": "10px 0px",
                "backgroundColor": "#C4DFDF",
            },
        )
