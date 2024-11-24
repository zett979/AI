import streamlit as st
import pandas as pd

st.set_page_config(page_title="Home", initial_sidebar_state="collapsed")

# BaseStyle()

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

@st.dialog("Cast your items")
def customize():
    df = st.session_state.uploaded_data
    st.write("Using column")
    for column in df.columns:
        st.text(column)


def main():
    st.write("Home page")
    st.write("Upload your dataset here:")
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file:
        try:
            # Store the uploaded file as a DataFrame in session state
            df = pd.read_csv(uploaded_file, index_col=1)
            st.session_state.uploaded_data = df
            st.success("File uploaded successfully!")
            # customize()
        except Exception as e:
            st.error(f"Error: {e}")

    st.write("Navigate to the **Data Profiling** page to analyze the uploaded data.")

main()