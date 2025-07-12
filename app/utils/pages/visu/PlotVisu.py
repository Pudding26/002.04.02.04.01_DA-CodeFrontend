import streamlit as st
import requests
import os
import sys



PLOTS = {
    "Barplot": {
        "Histogram": {
            "aggs": ["count"],
            "controls": {
                "col_facet": True,
                "row_facet": True,
                "hue": {"enabled": True, "max_hue_cols": 5}
            }
        }
    },
    "Lineplot": {
        "Density": {
            "aggs": ["mean", "sum"],
            "controls": {
                "col_facet": True,
                "row_facet": True,
                "hue": {"enabled": False}
            }
        }
    }
}


class PlotVisu:
    def __init__(self, df, shared_state=None):
        self.df = df
        self.shared_state = shared_state or {}

    def render(self):
        st.title("📈 Plot Configuration")

        plot_type = st.selectbox("Plot Type", list(PLOTS.keys()))
        subplots = list(PLOTS[plot_type].keys())
        subplot_type = st.selectbox("Subplot Type", subplots)

        subplot_config = PLOTS[plot_type][subplot_type]
        aggs = subplot_config["aggs"]
        agg_func = st.selectbox("Aggregation", aggs)

        col_facet = None
        row_facet = None
        hue_col = None
        max_hue_cols = None

        if subplot_config["controls"].get("col_facet"):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                col_facet = st.selectbox("Col Facet", self.df.columns)
            with col2:
                max_bins = st.selectbox("Max Bins", [5, 10, 20, 50], index=1)

        
        
        
        if subplot_config["controls"].get("row_facet"):
            row_facet = st.multiselect("Row Facet(s)", self.df.columns)

        hue_cfg = subplot_config["controls"].get("hue", {})
        if hue_cfg.get("enabled"):
            cols = st.columns([0.8, 0.2])
            with cols[0]:
                hue_col = st.selectbox("Hue Column", self.df.columns)
            with cols[1]:
                max_hue_cols = st.selectbox("Max Hue Categories", [2, 3, 4, 5], index=hue_cfg.get("max_hue_cols", 5) - 2)

        fast_mode = st.checkbox("⚡ Fast mode (sample 100 rows, low DPI)")

        agg_col = None
        if agg_func != "count":
            agg_col = st.selectbox("Aggregation Column", self.df.select_dtypes(include="number").columns)

        if st.button("Render Plot"):
            # Separate method for actual plotting
            self.handle_plot(
                plot_type, subplot_type, agg_func, agg_col,
                col_facet, row_facet, hue_col, max_hue_cols, fast_mode
            )

    def handle_plot(self, plot_type, subplot_type, agg_func, agg_col, col_facet, row_facet, hue_col, max_bins, fast_mode):
        df_prep = self.prepare_data(
            agg_func, agg_col, col_facet, row_facet, hue_col, max_bins, fast_mode
        )

        if plot_type == "Barplot":
            fig = self.plot_bar(df_prep, col_facet, row_facet, hue_col)
        elif plot_type == "Lineplot":
            fig = self.plot_line(df_prep, col_facet, row_facet, hue_col)
        else:
            st.error(f"Unsupported plot type: {plot_type}")
            return

        st.pyplot(fig)


    def prepare_data(self, agg_func, agg_col, col_facet, row_facet, hue_col, max_bins, fast_mode):
        import numpy as np
        import pandas as pd

        df = self.df.copy()

        # Ensure row_facet is a list
        row_facet = row_facet if row_facet else []

        # Columns needed:
        cols_needed = [col_facet] + row_facet
        if hue_col:
            cols_needed.append(hue_col)
        if agg_func != "count":
            cols_needed.append(agg_col)

        cols_needed = [c for c in cols_needed if c]  # Remove None
        df = df[cols_needed].copy()

        if fast_mode:
            df = df.sample(min(100, len(df)))

        # Handle col_facet binning if numeric
        if col_facet and pd.api.types.is_numeric_dtype(df[col_facet]):
            df[col_facet] = pd.cut(df[col_facet], bins=max_bins)

        # Group keys flattening
        group_keys = []
        if col_facet:
            group_keys.append(col_facet)
        group_keys.extend(row_facet)
        if hue_col:
            group_keys.append(hue_col)

        if agg_func == "count":
            df_grouped = df.groupby(group_keys).size().reset_index(name="value")
        else:
            df_grouped = df.groupby(group_keys)[agg_col].agg(agg_func).reset_index(name="value")

        return df_grouped


    def plot_bar(self, df, col_facet, row_facet, hue_col):
        import matplotlib.pyplot as plt

        dpi = 50
        fig, ax = plt.subplots(dpi=dpi)

        # Basic bar plot logic, could extend here for row/col facets too
        if hue_col:
            for name, grp in df.groupby(hue_col):
                ax.bar(grp[col_facet] if col_facet else grp.index, grp["value"], label=name)
            ax.legend(title=hue_col)
        else:
            ax.bar(df[col_facet] if col_facet else df.index, df["value"])

        return fig

    def plot_line(self, df, col_facet, row_facet, hue_col):
        import matplotlib.pyplot as plt

        dpi = 50
        fig, ax = plt.subplots(dpi=dpi)

        if hue_col:
            for name, grp in df.groupby(hue_col):
                ax.plot(grp[col_facet] if col_facet else grp.index, grp["value"], label=name)
            ax.legend(title=hue_col)
        else:
            ax.plot(df[col_facet] if col_facet else df.index, df["value"])

        return fig
