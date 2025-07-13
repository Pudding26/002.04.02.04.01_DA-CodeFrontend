import streamlit as st
import importlib
import pandas as pd

# Example DATA_SPECIFIC_CONFIG structure
DATA_SPECIFIC_CONFIG = {
    "production": {
        "modellingResults": {
            "histogram": {
                "allowed_row": ["knn_acc", "rf_acc"],
                "allowed_hue": ["species"],
                "allowed_col": ["scope"]
            },
            "density": {
                "allowed_row": ["feature1"],
                "allowed_hue": ["species"]
            }
        }
    }
}

PLOTS = {
    "Barplot": {
        "Histogram": {
            "controls": {
                "col_facet": True,
                "row_facet": True,
                "hue": {"enabled": False}
            }
        }
    },
    "Lineplot": {
        "Density": {
            "controls": {
                "col_facet": True,
                "row_facet": True,
                "hue": {"enabled": True}
            }
        },
        "Sum": {
            "controls": {
                "col_facet": True,
                "row_facet": True,
                "hue": {"enabled": True}
            }
        }
    }
}

class PlotVisu:
    def __init__(self, df, shared_state=None):
        """Initialize PlotVisu with dataframe and shared state."""
        self.df = df
        self.shared_state = shared_state or {}

    def render(self):
        """Render Streamlit UI for plot config and call handle_plot when user requests."""
        st.title("📈 Plot Configuration")

        plot_type = st.selectbox("Plot Type", list(PLOTS.keys()))
        subplot_type = st.selectbox("Subplot Type", list(PLOTS[plot_type].keys()))
        config = PLOTS[plot_type][subplot_type]["controls"]

        db_key = self.shared_state.get("db")
        table_name = self.shared_state.get("table")
        plot_key = subplot_type.lower()

        # Lookup dataset-specific config for this plot
        plot_cfg = DATA_SPECIFIC_CONFIG.get(db_key, {}).get(table_name, {}).get(plot_key, {})

        all_cols = list(self.df.columns)
        row_cols = [c for c in all_cols if "allowed_row" not in plot_cfg or c in plot_cfg["allowed_row"]]
        hue_cols = [c for c in all_cols if "allowed_hue" not in plot_cfg or c in plot_cfg["allowed_hue"]]
        col_cols = [c for c in all_cols if "allowed_col" not in plot_cfg or c in plot_cfg["allowed_col"]]

        col_facet = None
        row_facet = []
        hue_col = None
        max_bins = 10
        max_col_bins = 5

        if config.get("col_facet"):
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                col_facet = st.selectbox("Col Facet", col_cols)
            with col2:
                max_col_bins = st.number_input("Max Col Bins", min_value=2, max_value=20, value=5)

        if config.get("row_facet"):
            row_facet = st.multiselect("Row Facet(s)", row_cols)

        if config.get("hue", {}).get("enabled"):
            hue_col = st.selectbox("Hue Column", hue_cols)

        if subplot_type == "Histogram":
            max_bins = st.selectbox("Max Bins", [5, 10, 20, 50], index=1)

        fast_mode = st.checkbox("⚡ Fast mode (sample 100 rows)")
        multi_plot = st.checkbox("📄 Multi-plot (single grid figure)", value=True)

        if st.button("Render Plot"):
            self.handle_plot(
                plot_type, subplot_type,
                col_facet, row_facet, hue_col,
                max_bins, fast_mode, multi_plot,
                max_col_bins
            )

    def handle_plot(self, plot_type, subplot_type,
                    col_facet, row_facet, hue_col,
                    max_bins, fast_mode, multi_plot,
                    max_col_bins):
        """Prepare data and dynamically call appropriate plot function."""
        df_prep = self.prepare_data(
            col_facet, row_facet, hue_col, fast_mode, max_col_bins
        )

        module_name = f"utils.pages.visu.plots.plot_{plot_type.lower()}"
        plot_func_name = f"plot_{subplot_type.lower()}"

        try:
            plot_module = importlib.import_module(module_name)
            plot_func = getattr(plot_module, plot_func_name)
        except (ImportError, AttributeError) as e:
            st.error(f"Plot function not found: {e}")
            return

        result = plot_func(df_prep, col_facet, row_facet, hue_col, max_bins, multi_plot)

        if isinstance(result, tuple):
            figs, num_cols = result
        else:
            figs = result
            num_cols = None

        if isinstance(figs, list):
            if num_cols is None or num_cols <= 1:
                # Fallback: render vertically
                for f in figs:
                    st.pyplot(f)
            else:
                rows = [figs[i:i+num_cols] for i in range(0, len(figs), num_cols)]
                for row_figs in rows:
                    cols = st.columns(len(row_figs))
                    for col, fig in zip(cols, row_figs):
                        with col:
                            st.pyplot(fig)
        elif figs:
            st.pyplot(figs)


    def prepare_data(self, col_facet, row_facet, hue_col, fast_mode, max_col_bins):
        """Prepare dataframe by sampling, selecting cols and reducing col_facet."""
        df = self.df.copy()

        if fast_mode:
            df = df.sample(min(100, len(df)))

        cols_needed = set()
        if col_facet:
            cols_needed.add(col_facet)
        if row_facet:
            cols_needed.update(row_facet)
        if hue_col:
            cols_needed.add(hue_col)

        if cols_needed:
            df = df[list(cols_needed)].copy()

        if col_facet:
            df = self.reduce_col_facet(df, col_facet, max_col_bins)

        return df

    def reduce_col_facet(self, df, col_facet, max_bins):
        """Reduce col_facet to max_bins unique values + 'Rest' for rare classes."""
        if col_facet not in df.columns:
            return df

        if pd.api.types.is_numeric_dtype(df[col_facet]):
            df[col_facet] = pd.qcut(df[col_facet], q=max_bins, duplicates='drop')
        else:
            top_cats = df[col_facet].value_counts().nlargest(max_bins).index
            df[col_facet] = df[col_facet].apply(lambda x: x if x in top_cats else "Rest")

        return df
