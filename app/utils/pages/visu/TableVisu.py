"""
TableVisu

Interactive tabular visualization.

Features:
  - Select which columns to display
  - Show table preview with Streamlit Data Editor
  - Export filtered table as CSV
  - Persist visible columns via central state
"""

import streamlit as st
from app.utils.pages.visu.BaseVisu import BaseVisu


class TableVisu(BaseVisu):
    """
    Visualization class for rendering interactive tables.

    State contract:
        initial_config (dict):
            {
                "cols": ["col1", "col2", ...]   # visible columns
            }

        get_state() -> dict:
            {
                "cols": ["col1", "col2", ...]   # visible columns
            }
    """

    def __init__(self, df, shared_state, initial_config=None):
        """
        Initialize the TableVisu.

        Args:
            df (pd.DataFrame): The dataframe to visualize.
            shared_state (dict): Shared info (db, table name, etc.).
            initial_config (dict, optional): Previously stored state.
        """
        super().__init__(df, shared_state, initial_config)

        default_cols = self.df.columns.tolist()
        init_cols = self.initial_config.get("cols", default_cols)

        # Ensure visible_cols is in session_state
        if "visible_cols" not in st.session_state:
            st.session_state.visible_cols = init_cols

    # -------------------------------
    # State Handling
    # -------------------------------
    def get_state(self):
        """Return the current visualization state (visible columns)."""
        return {"cols": st.session_state.get("visible_cols", [])}

    # -------------------------------
    # Rendering
    # -------------------------------
    def render(self):
        """Render the interactive table with column selection and download option."""
        st.subheader("📊 Interactive Table")

        # Split into two columns: selector (1/5) and table view (4/5)
        selector_col, table_col = st.columns([1, 5])

        # --- Column selector UI ---
        with selector_col:
            st.markdown("### 🔧 Columns")
            with st.container(height=300, border=True):
                for col in self.df.columns:
                    checked = col in st.session_state.visible_cols
                    if st.checkbox(col, value=checked, key=f"col_chk_{col}"):
                        if col not in st.session_state.visible_cols:
                            st.session_state.visible_cols.append(col)
                    else:
                        if col in st.session_state.visible_cols:
                            st.session_state.visible_cols.remove(col)

        # --- Table preview + download ---
        with table_col:
            selected_cols = [c for c in self.df.columns if c in st.session_state.visible_cols]

            if not selected_cols:
                st.warning("⚠️ Please select at least one column to display.")
                return

            # Table preview
            st.data_editor(
                self.df[selected_cols],
                use_container_width=True,
                disabled=True,
                hide_index=True
            )

            # Download filtered table
            st.download_button(
                "📥 Download CSV",
                data=self.df[selected_cols].to_csv(index=False),
                file_name="filtered_table.csv",
                mime="text/csv"
            )
