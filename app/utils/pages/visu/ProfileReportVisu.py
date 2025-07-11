import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report  # ✅ new import
from app.utils.pages.visu.BaseVisu import BaseVisu


class ProfileReportVisu(BaseVisu):
    def __init__(self, df, shared_state, initial_config=None):
        super().__init__(df, shared_state, initial_config)

    def get_state(self):
        return {}

    def render(self):
        st.subheader("📋 Data Profile Summary")

        with st.spinner("Generating profile report..."):
            profile = ProfileReport(
                self.df,
                title="Data Profiling Report",
                explorative=True,
            )

            # ✅ Use streamlit-ydata-profiling to render the report
            st_profile_report(profile, navbar=True)
