import streamlit as st
from components.Base import BaseStyle, Theme, ButtonStyle
from streamlit_elements import elements, mui, html

st.set_page_config(page_title="Analysis")

BaseStyle()


def main():
    st.write("Analysis Page")
    with elements("multiple_children"):
        mui.Button(
            "Button",
            sx=ButtonStyle(ghost=False, size="sm", variant="primary"),
            elevation=0,
            disableRipple=True,
            className="sample",
        )


main()
