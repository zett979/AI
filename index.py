import streamlit as st
import pandas as pd
from components.Base import BaseStyle

st.set_page_config(page_title="Home", initial_sidebar_state="collapsed")

# st.set_page_config(page_title="Analysis")

# BaseStyle()

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

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
        except Exception as e:
            st.error(f"Error: {e}")

    st.write("Navigate to the **Data Profiling** page to analyze the uploaded data.")

main()
