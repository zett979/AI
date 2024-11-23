import streamlit as st
from ydata_profiling import ProfileReport
from components.Base import BaseStyle, Theme, ButtonStyle

st.set_page_config(page_title="Data Profiling")

# BaseStyle()

def main():
    st.title("Data Profiling")

    # Checking if data is available in session state
    if "uploaded_data" in st.session_state:
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            st.write("Dataset loaded from the home page")
            st.dataframe(df)

            st.write("Generating data profiling report...")
            profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
            
            # Save the report as HTML and display it
            profile.to_file("profile_report.html")

            with open("profile_report.html", "r") as f:
                st.components.v1.html(f.read(), height=1000, scrolling=True)

            # Converting the profiling report to HTML for download
            report_html = profile.to_html()

            # Add a download button for the report
            st.download_button(
                label="Download Full Report",
                data=report_html,
                file_name="data_profiling_report.html",
                mime="text/html"
            )

        else:
            st.warning("No data found. Please upload a dataset on the Home page.")
    else:
        st.warning("Session state is empty. Please upload a dataset on the Home page.")

main()