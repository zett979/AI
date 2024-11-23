import streamlit as st
import toml


def BaseStyle():
    with open("./components/app.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data
def Theme():
    with open("theme.toml", "r") as f:
        config = toml.load(f)
    return config["themes"]


@st.cache_data
def ButtonStyle(variant="primary", size="sm", ghost=False):
    """
    Get the button style including hover and disabled for each variant and sizes
    Args:
        variant: (primary(default), secondary, success, warning, error)
        size: md(default), sm
        ghost: True, False
    """
    sx = {
        "border": "0px",
        "boxShadow": "none",
        "backgroundColor": "transparent",
        "color": Theme()["text"],
        "textTransform": "capitalize",
    }
    if ghost == True:
        sx["&:disabled"] = {"color": Theme()["disabled"]}
    else:
        sx["&:disabled"] = {"color": Theme()["disabledGhost"]}
    if variant == "primary":
        if ghost == False:
            sx["backgroundColor"] = Theme()["primary"]
            sx["&:hover"] = {"backgroundColor": Theme()["primaryHover"]}
            sx["&:disabled"] = {"backgroundColor": Theme()["primaryDisabled"]}
        else:
            sx["&:hover"] = {"backgroundColor": Theme()["primaryLight"]}
            sx["border"] = "1px solid " + Theme()["primary"]

    elif variant == "secondary":
        if ghost == False:
            sx["backgroundColor"] = Theme()["secondary"]
            sx["&:hover"] = {"backgroundColor": Theme()["secondaryHover"]}
            sx["&:disabled"] = {"backgroundColor": Theme()["secondaryDisabled"]}
        else:
            sx["&:hover"] = {"backgroundColor": Theme()["secondaryLight"]}
            sx["border"] = "1px solid " + Theme()["secondary"]

    elif variant == "success":
        if ghost == False:
            sx["backgroundColor"] = Theme()["success"]
            sx["&:hover"] = {"backgroundColor": Theme()["successHover"]}
            sx["&:disabled"] = {"backgroundColor": Theme()["successDisabled"]}
        else:
            sx["&:hover"] = {"backgroundColor": Theme()["successLight"]}
            sx["border"] = "1px solid " + Theme()["success"]

    elif variant == "warning":
        if ghost == False:
            sx["backgroundColor"] = Theme()["warning"]
            sx["&:hover"] = {"backgroundColor": Theme()["warningHover"]}
            sx["&:disabled"] = {"backgroundColor": Theme()["warningDisabled"]}
        else:
            sx["&:hover"] = {"backgroundColor": Theme()["warningLight"]}
            sx["border"] = "1px solid " + Theme()["warning"]
    elif variant == "error":
        if ghost == False:
            sx["backgroundColor"] = Theme()["error"]
            sx["&:hover"] = {"backgroundColor": Theme()["errorHover"]}
            sx["&:disabled"] = {"backgroundColor": Theme()["errorDisabled"]}
        else:
            sx["&:hover"] = {"backgroundColor": Theme()["errorLight"]}
            sx["border"] = "1px solid " + Theme()["error"]

    # default spacing is n * 8 px
    if size == "sm":
        sx["p"] = 1.25
        sx["borderRadius"] = "8px"
        sx["fontSize"] = "16px"
    else:
        sx["px"] = 2.5
        sx["py"] = 2
        sx["borderRadius"] = "10px"
        sx["fontSize"] = "20px"
        sx["fontWeight"] = "semibold"

    return sx
