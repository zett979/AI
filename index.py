import streamlit as st
from components.Base import BaseStyle

st.set_page_config(page_title="Home", initial_sidebar_state="collapsed")

st.set_page_config(page_title="Analysis")

BaseStyle()


def main():
    st.write("Home page")
