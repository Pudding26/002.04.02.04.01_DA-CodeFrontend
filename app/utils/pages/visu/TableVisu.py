import streamlit as st
from app.utils.pages.visu.BaseVisu import BaseVisu


class TableVisu(BaseVisu):
    def __init__(self, df, shared_state, initial_config=None):
        super().__init__(df, shared_state, initial_config)

        # DEBUG — show what config is received
        if st.session_state.get("debug_once") is None:
            st.session_state.debug_once = True
            #st.write("🔍 Initial config received:", self.initial_config)

        default_cols = self.df.columns.tolist()
        init_cols = self.initial_config.get("cols", default_cols)

        if "visible_cols" not in st.session_state:
            st.session_state.visible_cols = init_cols


    def get_state(self):
        return {
            "cols": st.session_state.get("visible_cols", [])
        }

    def render(self):
        st.subheader("📊 Interactive Table")

        col_selector_col, table_col = st.columns([1, 5])

        with col_selector_col:
            st.markdown("### 🔧 Columns")
            with st.container(height = 300, border=True):
                for col in self.df.columns:
                    checked = col in st.session_state.visible_cols
                    if st.checkbox(col, value=checked, key=f"col_chk_{col}"):
                        if col not in st.session_state.visible_cols:
                            st.session_state.visible_cols.append(col)
                    else:
                        if col in st.session_state.visible_cols:
                            st.session_state.visible_cols.remove(col)

        with table_col:
            selected_cols = [col for col in self.df.columns if col in st.session_state.visible_cols]
            if not selected_cols:
                st.warning("Please select at least one column.")
                return

            st.data_editor(
                self.df[selected_cols],
                use_container_width=True,
                disabled=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download CSV",
                data=self.df[selected_cols].to_csv(index=False),
                file_name="filtered_table.csv",
                mime="text/csv"
            )
