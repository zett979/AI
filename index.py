import streamlit as st
from components.Base import BaseStyle
from components.Navbar import NavBar

st.set_page_config(page_title="Home", initial_sidebar_state="collapsed")


BaseStyle()


def main():
    st.write("Home page")
