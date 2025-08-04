"""
PlotVisu

Configurable plotting interface with support for barplots, lineplots, histograms, etc.

Integration of three YAML configs:
- plot_config.yaml: defines plot types & controls
- plot_control_config.yaml: reusable control templates
- plot_data_specific_config.yaml: dataset-specific restrictions
"""

import streamlit as st
import pandas as pd
import logging
import yaml
import os
from app.utils.pages.visu.BaseVisu import BaseVisu
from app.utils.pages.visu.utils.render_control import render_control

# Logging setup
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# -------------------------------
# Config loading
# -------------------------------

def load_configs():
    with open("config/plot_config.yaml", "r") as f:
        plot_config = yaml.safe_load(f)
    with open("config/plot_control_config.yaml", "r") as f:
        control_config = yaml.safe_load(f)
    with open("config/plot_data_specific_config.yaml", "r") as f:
        data_specific_config = yaml.safe_load(f).get("data_specific_config", {})
    return plot_config, control_config, data_specific_config

plot_config, CONTROL_CONFIG, DATA_SPECIFIC_CONFIG = load_configs()
PLOTS = plot_config.get("plots", {})


# -------------------------------
# Control resolver
# -------------------------------
def resolve_controls(control_specs):
    """
    Merge plot-specific control specs with global control templates.

    Rules:
    - If control spec is a dict, merge with template from CONTROL_CONFIG (if exists).
    - If control spec is a string referencing a template, resolve it.
    - Otherwise, return as-is.
    """
    resolved = {}
    for name, spec in control_specs.items():
        if isinstance(spec, str) and spec in CONTROL_CONFIG:
            resolved[name] = CONTROL_CONFIG[spec]
        elif isinstance(spec, dict):
            base = CONTROL_CONFIG.get(name, {})
            merged = {**base, **spec}  # plot-specific overrides global template
            resolved[name] = merged
        elif name in CONTROL_CONFIG:
            resolved[name] = CONTROL_CONFIG[name]
        else:
            resolved[name] = spec
    return resolved


# -------------------------------
# Main Class
# -------------------------------
class PlotVisu(BaseVisu):
    def __init__(self, df, shared_state=None, initial_config=None):
        super().__init__(df, shared_state, initial_config)
        self.config = initial_config or {}

    def get_state(self):
        """Return current plot configuration for persistence in URL."""
        return self.config

    def render(self):
        st.subheader("📈 Plot Configuration")
        plot_config, CONTROL_CONFIG, DATA_SPECIFIC_CONFIG = load_configs()
        PLOTS = plot_config.get("plots", {})
        # --- Row 1: Plot type & Subplot type ---
        c1, c2 = st.columns(2)
        with c1:
            plot_type = st.selectbox(
                "Plot Type",
                list(PLOTS.keys()),
                index=list(PLOTS.keys()).index(self.config.get("plot_type", list(PLOTS.keys())[0]))
            )
        with c2:
            subplot_options = list(PLOTS[plot_type].keys())
            subplot_type = st.selectbox(
                "Subplot Type",
                subplot_options,
                index=subplot_options.index(self.config.get("subplot_type", subplot_options[0]))
            )

        self.config["plot_type"] = plot_type
        self.config["subplot_type"] = subplot_type

        # --- Control specs from YAML (resolved with templates) ---
        controls = PLOTS[plot_type][subplot_type]["controls"]

        # --- Dataset-specific config ---
        db_key = self.shared_state.get("db")
        table_name = self.shared_state.get("table")
        plot_key = subplot_type.lower()
        dataset_cfg = DATA_SPECIFIC_CONFIG.get(db_key, {}).get(table_name, {})
        category_order = dataset_cfg.get("category_order", {})
        data_cfg = next((v for k, v in dataset_cfg.items() if k.lower() == plot_key), {})

        # --- Allowed columns ---
        all_cols = list(self.df.columns)
        numeric_cols = list(self.df.select_dtypes(include="number").columns)

        def filter_allowed(cols, allowed):
            return [c for c in cols if allowed is None or c.lower() in [a.lower() for a in allowed]]

        allowed_cols = {
            "row": filter_allowed(all_cols, data_cfg.get("allowed_row")),
            "col": filter_allowed(all_cols, data_cfg.get("allowed_col")),
            "hue": filter_allowed(all_cols, data_cfg.get("allowed_hue")),
            "weight": filter_allowed(all_cols, data_cfg.get("allowed_weight")),
            "numeric": numeric_cols,
        }

        # --- Render each control using central function ---
        for name, spec in controls.items():
            help_text = spec.get("doc")   # 👈 read inline doc from YAML
            val = render_control(
                name, spec, self.df, allowed_cols,
                self.config.get(name), key_prefix=subplot_type,
                help=help_text   # 👈 pass it along
            )

            # Normalize "None" → None
            if spec.get("type") in ["select", "multiselect"]:
                if val == "None":
                    val = None
                elif isinstance(val, list) and "None" in val:
                    val = [v for v in val if v != "None"]

            self.config[name] = val

        custom_cfg = data_cfg.get("customization", {})
        if custom_cfg:
            with st.expander("⚙️ Customization"):
                controls_per_row = 5
                items = list(custom_cfg.items())

                for start in range(0, len(items), controls_per_row):
                    row_items = items[start:start + controls_per_row]
                    cols = st.columns(len(row_items))
                    for (name, spec), col in zip(row_items, cols):
                        with col:
                            help_text = spec.get("doc")
                            val = render_control(
                                name, spec, self.df, allowed_cols,
                                self.config.get(name),
                                key_prefix=f"{subplot_type}_custom",
                                help=help_text
                            )
                            self.config[name] = val



        # --- Render button ---
        if st.button("Render Plot"):
            self.handle_plot(category_order)

    # -------------------------------
    # Plotting
    # -------------------------------
    def handle_plot(self, category_order):
        cfg = self.config
        df_prep = self.prepare_data(
            cfg.get("col_facet"),
            cfg.get("row_facet"),
            cfg.get("hue_col"),
            cfg.get("fast_mode", False),
            cfg.get("max_col_bins", 5),
            cfg.get("agg_col"),
            cfg.get("weight_col"),
            cfg.get("x_col"),
            cfg.get("y_col"),
        )
        st.write("🔧 Debug config", cfg)

        init_kwargs = {
            'df': df_prep,
            'col_facet': cfg.get("col_facet"),
            'row_facet': cfg.get("row_facet"),
            'hue_col': cfg.get("hue_col"),
            'category_order': category_order
        }

        # Dispatch plotting by subplot type
        if cfg["subplot_type"] == "GroupedBarplot":
            from app.utils.pages.visu.plots.plot_barplot import GroupedBarplot
            plot_instance = GroupedBarplot(**init_kwargs)
            figs = plot_instance.plot(
                agg_func=cfg.get("agg_func", "count"),
                agg_col=cfg.get("agg_col"),
                weighting_mode=cfg.get("weighting_mode"),
                weight_col=cfg.get("weight_col"),
                max_bins=cfg.get("max_bins", 10),
                multi_plot=cfg.get("multi_plot", True)
            )

        elif cfg["subplot_type"] == "Histogram":
            from utils.pages.visu.plots.barplot.HistogramBarplot import HistogramBarplot
            plot_instance = HistogramBarplot(**init_kwargs)
            figs = plot_instance.plot(
                max_bins=cfg.get("max_bins", 10),
                multi_plot=cfg.get("multi_plot", True),
                plot_mode=cfg.get("plot_mode", "bars"),
                value_range=cfg.get("value_range")
            )

        elif cfg["subplot_type"] == "LineSumPlot":
            from utils.pages.visu.plots.plot_lineplot import LineSumPlot
            plot_instance = LineSumPlot(**init_kwargs)
            figs = plot_instance.plot(
                weighting_mode=cfg.get("weighting_mode"),
                weight_col=cfg.get("weight_col"),
                multi_plot=cfg.get("multi_plot", True),
                agg_col=cfg.get("agg_col")
            )

        elif cfg["subplot_type"] == "LineDensityPlot":
            from utils.pages.visu.plots.plot_lineplot import DensityPlot
            plot_instance = DensityPlot(**init_kwargs)
            figs = plot_instance.plot(
                agg_cols=cfg.get("agg_col"),
                weighting_mode=cfg.get("weighting_mode"),
                weight_col=cfg.get("weight_col"),
                multi_plot=cfg.get("multi_plot", True)
            )

        elif cfg["subplot_type"] == "SimpleLineplot":
            from app.utils.pages.visu.plots.lineplot.SimpleLineplot import SimpleLineplot
            plot_instance = SimpleLineplot(**init_kwargs)
            figs = plot_instance.plot(
                x_col=cfg.get("x_col"),
                y_col=cfg.get("y_col"),
                multi_plot=cfg.get("multi_plot", True)
            )
        else:
            figs = None

        # Render results
        if isinstance(figs, tuple):
            figs, num_cols = figs
        else:
            num_cols = None

        if isinstance(figs, list):
            if num_cols is None or num_cols <= 1:
                for f in figs:
                    st.pyplot(f)
            else:
                rows = [figs[i:i+num_cols] for i in range(0, len(figs), num_cols)]
                for row_figs in rows:
                    cols = st.columns(len(row_figs))
                    for col, fig in zip(cols, row_figs):
                        with col:
                            st.pyplot(fig)
        elif figs is not None:
            st.pyplot(figs)

    # -------------------------------
    # Data Prep
    # -------------------------------
    def prepare_data(self, col_facet, row_facet, hue_col, fast_mode, max_col_bins,
                    agg_col=None, weight_col=None, x_col=None, y_col=None):
        df = self.df.copy()

        if fast_mode and len(df) > 100:
            df = df.sample(100)

        cols_needed = set()
        requested = set()

        if col_facet:
            requested.add(col_facet)
        if row_facet:
            requested.update(row_facet if isinstance(row_facet, list) else [row_facet])
        if hue_col:
            requested.add(hue_col)
        if isinstance(agg_col, list):
            requested.update([c for c in agg_col if c])
        elif agg_col:
            requested.add(agg_col)
        if weight_col:
            requested.add(weight_col)
        if x_col:
            requested.add(x_col)
        if y_col:
            requested.add(y_col)

        # Keep only those actually present in df
        valid_cols = [c for c in requested if c in df.columns]
        missing_cols = [c for c in requested if c not in df.columns]

        if missing_cols:
            logging.warning(f"[prepare_data] Missing columns (dropped): {missing_cols}")

        if not valid_cols:
            logging.warning("[prepare_data] No valid columns left → returning empty DataFrame")
            return pd.DataFrame()

        df = df[valid_cols].copy()

        if col_facet and col_facet in df.columns:
            df = self.reduce_col_facet(df, col_facet, max_col_bins)

        logging.info(f"[prepare_data] Returning df shape={df.shape}, cols={df.columns.tolist()}")
        return df




    def reduce_col_facet(self, df, col_facet, max_bins):
        if col_facet not in df.columns:
            return df
        if pd.api.types.is_numeric_dtype(df[col_facet]):
            df[col_facet] = pd.qcut(df[col_facet], q=max_bins, duplicates='drop')
        else:
            top_cats = df[col_facet].value_counts().nlargest(max_bins).index
            df[col_facet] = df[col_facet].apply(lambda x: x if x in top_cats else "Rest")
        return df
