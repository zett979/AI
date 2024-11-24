import streamlit as st
import pandas as pd
from components.Base import BaseStyle
from streamlit_elements import elements, mui, html

st.set_page_config(page_title="Home", initial_sidebar_state="collapsed")

BaseStyle()

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

if "isRowMain" not in st.session_state:
    st.session_state.isRowMain = False

@st.dialog("Cast your items")
def customize(df):
    st.write(f"Choose your items")
    on = st.toggle('Use row')
    if on:
        st.write("Using row")
        for index, row in df.iterrows():
            st.write(row.name)
    else: 
        st.write("Using column")
        for column in df.columns:
            st.text(column)
    st.session_state.isRowMain = on


def main():
    st.write("Home page")
    st.write("Upload your dataset here:")
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file:
        try:
            # Store the uploaded file as a DataFrame in session state
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.success("File uploaded successfully!")
            customize(df)
        except Exception as e:
            st.error(f"Error: {e}")

    st.write("Navigate to the **Data Profiling** page to analyze the uploaded data.")

main()