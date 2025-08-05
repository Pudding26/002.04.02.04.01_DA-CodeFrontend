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
logger = logging.getLogger("plotvisu")


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
        import pandas as pd
        import logging
        logger = logging.getLogger("app.utils.pages.visu.PlotVisu")

        st.subheader("📈 Plot Configuration")
        plot_config, CONTROL_CONFIG, DATA_SPECIFIC_CONFIG = load_configs()
        PLOTS = plot_config.get("plots", {})

        # --- Row 1: Plot type & Subplot type ---
        c1, c2 = st.columns(2)
        with c1:
            plot_type = st.selectbox(
                "Plot Type",
                list(PLOTS.keys()),
                index=list(PLOTS.keys()).index(
                    self.config.get("plot_type", list(PLOTS.keys())[0])
                )
            )
        with c2:
            subplot_options = list(PLOTS[plot_type].keys())
            default_subplot = self.config.get("subplot_type")
            if default_subplot not in subplot_options:
                logger.info(f"[render] Resetting subplot_type → {subplot_options[0]} (was {default_subplot})")
                default_subplot = subplot_options[0]

            subplot_type = st.selectbox(
                "Subplot Type",
                subplot_options,
                index=subplot_options.index(default_subplot)
            )

        # Reset config if plot_type changed
        if plot_type != self.config.get("plot_type"):
            logger.info(f"[render] Plot type changed → resetting config")
            self.config.clear()
            self.config["plot_type"] = plot_type

        self.config["plot_type"] = plot_type
        self.config["subplot_type"] = subplot_type
        logger.info(f"[render] Selected plot_type={plot_type}, subplot_type={subplot_type}")

        # --- Control specs from YAML ---
        controls = PLOTS[plot_type][subplot_type]["controls"]
        logger.debug(f"[render] Loaded controls: {list(controls.keys())}")

        # --- Dataset-specific config ---
        db_key = self.shared_state.get("db")
        table_name = self.shared_state.get("table")
        dataset_cfg = DATA_SPECIFIC_CONFIG.get(db_key, {}).get(table_name, {}) or {}
        plot_key = subplot_type.lower()
        data_cfg = next((v for k, v in dataset_cfg.items() if k.lower() == plot_key), {})
        category_order = dataset_cfg.get("category_order", {}) or {}

        logger.debug(f"[render] db={db_key}, table={table_name}, "
                    f"dataset_cfg keys={list(dataset_cfg.keys())}, "
                    f"category_order={category_order}")

        
        # --- Allowed columns (cached) ---
        allowed_cols = get_allowed_cols(self.df, data_cfg)
        logger.debug(f"[render] allowed_cols: {allowed_cols}")


        # --- Helper for dropdown + extras in one row ---

        def render_col_with_extras(name, spec, base, subplot_type):
            options = allowed_cols.get(base, [])
            current_val = self.config.get(name)
            if current_val not in options and options:
                current_val = options[0]

            if not options:
                st.warning(f"No options available for {name}")
                return

            if base in ["x", "hue", "col"]:
                if current_val and current_val in self.df.columns:
                    series = self.df[current_val]

                    # --- Layout: 4 columns ---
                    c1, c2, c3, c4 = st.columns([1, 1, 3, 1])

                    # --- Column dropdown ---
                    with c1:
                        val = st.segmented_control(
                            spec.get("doc", name),
                            options=options,
                            default=current_val if current_val in options else options[0],  # ✅
                            key=f"{subplot_type}_{name}",
                            help=spec.get("doc"),
                            selection_mode="single",
                            width="stretch"
                        )


                    # --- Numeric case ---
                    if pd.api.types.is_numeric_dtype(series):
                        with c2:
                            use_bins = st.checkbox(
                                "Use binning?",
                                value=True,
                                key=f"{subplot_type}_{base}_binning_toggle"
                            )
                        if use_bins:
                            with c3:
                                bin_num = st.slider(
                                    "Bins",
                                    min_value=2, max_value=10,
                                    value=5,
                                    key=f"{subplot_type}_{base}_bin_auto"
                                )
                                self.config[f"percentile_bin_number_{base}"] = int(bin_num)
                        else:
                            with c3:
                                # 🚨 clear binning config when disabled
                                self.config.pop(f"percentile_bin_number_{base}", None)


                                unique_vals = get_unique_values(self.df, val)
                                sel_vals = st.segmented_control(
                                    "Values",
                                    selection_mode="multi",
                                    width="stretch", 
                                    options=unique_vals,
                                    default=unique_vals,
                                    key=f"{subplot_type}_{base}_values_auto"
                                )
                                # normalize to list
                                if isinstance(sel_vals, str):
                                    sel_vals = [sel_vals]
                                elif sel_vals is None:
                                    sel_vals = []
                                if not sel_vals:  # fallback to all
                                    sel_vals = unique_vals
                                print(f"[NUM]: for {base} Adding {sel_vals}")


                                self.config[f"{base}_values"] = sel_vals


                    # --- Categorical case ---
                    else:
                        with c2:
                            bin_rest = st.checkbox(
                                "Bin rest?",
                                value=False,
                                key=f"{subplot_type}_{base}_bin_rest_auto"
                            )
                            self.config[f"bin_rest_{base}"] = bin_rest
                        with c3:
                            unique_vals = get_unique_values(self.df, val)
                            sel_vals = st.segmented_control(
                                spec.get("doc", name),
                                options = unique_vals,
                                default=unique_vals[0],
                                key=f"{subplot_type}_{name}_auto",
                                help=spec.get("doc"),
                                selection_mode="multi",
                                width="stretch"
                            )
                            print(f"[CAT]: Adding {sel_vals}")
                            self.config[f"{base}_values"] = sel_vals if isinstance(sel_vals, list) else [sel_vals]

                    self.config[name] = val
                    return





        # --- Render controls in fixed order ---
        col_order = ["col_facet", "hue_col", "x_col"]
        ordered_controls = (
            [(k, v) for k, v in controls.items() if k in col_order] +
            [(k, v) for k, v in controls.items() if k not in col_order]
        )

        for name, spec in ordered_controls:
            base = name.split("_")[0]

            if name in ["col_facet", "hue_col", "x_col"]:
                render_col_with_extras(name, spec, base, subplot_type)
            else:
                val = render_control(
                    name, spec, self.df, allowed_cols,
                    self.config.get(name),
                    key_prefix=subplot_type,
                    help=spec.get("doc")
                )
                self.config[name] = val


        # --- Customization expander ---
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
                            logger.info(f"[render] Custom control {name} → {val}")

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

        elif cfg["subplot_type"] == "SimpleBoxplot":
            from app.utils.pages.visu.plots.boxplot.SimpleBoxplot import SimpleBoxplot
            plot_instance = SimpleBoxplot(**init_kwargs)
            figs = plot_instance.plot(
                multi_plot=cfg.get("multi_plot", True),
                x_col=cfg.get("x_col")
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
                    
        def apply_category_filter(df, col, values, bin_rest=False):
            if col and col in df.columns and values is not None:
                # normalize to list
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, (list, tuple, pd.Series)):
                    values = [values]

                before = len(df)
                if bin_rest:
                    df.loc[~df[col].isin(values), col] = "rest"
                    after = len(df)
                    logging.info(f"[apply_category_filter] col={col}, bin_rest=True, keep={values}, "
                                f"rows before={before}, after={after}")
                else:
                    df = df[df[col].isin(values)]
                    after = len(df)
                    logging.info(f"[apply_category_filter] col={col}, bin_rest=False, keep={values}, "
                                f"rows before={before}, after={after}")
            return df

        if fast_mode and len(df) > 100:
            df = df.sample(100)
            logging.info(f"[prepare_data] fast_mode=True, sampled 100 rows, now shape={df.shape}")

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
        logging.info(f"[prepare_data] Initial subset shape={df.shape}, cols={df.columns.tolist()}")

        # Apply col_facet reduction
        if col_facet and col_facet in df.columns:
            before = len(df)
            df = self.reduce_col_facet(df, col_facet, max_col_bins)
            after = len(df)
            logging.info(f"[prepare_data] reduce_col_facet on {col_facet}, rows before={before}, after={after}")

        # Apply filters for x_col, hue_col, col_facet
        for col_key in ["x_col", "hue_col", "col_facet"]:
            col = locals().get(col_key)
            if not col or col not in df.columns:
                continue

            base = col_key.split("_")[0]

            # Numeric: bin into percentiles if requested
            bin_key = f"percentile_bin_number_{base}"
            n_bins = self.config.get(bin_key)

            if n_bins:
                before = len(df)
                df = _apply_percentile_binning(df, col, n_bins)
                after = len(df)
                logger.info(f"[prepare_data] percentile binning col={col}, n_bins={n_bins}, rows before={before}, after={after}")

                # 🚫 Skip filtering if binning is active
                continue

            # Categorical (or unbinned numeric): apply value filtering + bin_rest
            values_key = f"{base}_values"
            bin_rest_key = f"bin_rest_{base}"
            values = self.config.get(values_key)
            bin_rest = self.config.get(bin_rest_key, False)
            df = apply_category_filter(df, col, values, bin_rest)


        logging.info(f"[prepare_data] Final df shape={df.shape}, cols={df.columns.tolist()}")
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

def _apply_percentile_binning(df, col, n_bins):
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        labels = [f"P{i+1}" for i in range(n_bins)]
        df[col] = pd.qcut(df[col], q=n_bins, labels=labels, duplicates="drop")
    return df


@st.cache_data
def get_allowed_cols(df, data_cfg):
    all_cols = list(df.columns)
    numeric_cols = list(df.select_dtypes(include="number").columns)

    def filter_allowed(cols, allowed):
        return [c for c in cols if allowed is None or c.lower() in [a.lower() for a in allowed]]

    return {
        "row": filter_allowed(all_cols, data_cfg.get("allowed_row")),
        "col": filter_allowed(all_cols, data_cfg.get("allowed_col")),
        "x": filter_allowed(all_cols, data_cfg.get("allowed_x", all_cols)),  # separate group
        "hue": filter_allowed(all_cols, data_cfg.get("allowed_hue")),
        "weight": filter_allowed(all_cols, data_cfg.get("allowed_weight")),
        "numeric": numeric_cols,
    }

@st.cache_data
def get_unique_values(df, col):
    return sorted(df[col].dropna().unique().tolist())
