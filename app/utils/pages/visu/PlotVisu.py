import streamlit as st
import importlib
import pandas as pd
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

DATA_SPECIFIC_CONFIG = {
    "production": {
        "modellingResults": {
            "category_order": {
                "scope": ["family", "genus", "species", "sourceID", "specimenID", "sampleID", "stackID", "shotID"],
                "label": ["family", "genus", "species", "sourceID", "specimenID", "sampleID", "stackID", "shotID"],
            },
            "Histogram": {
                "allowed_row": ["knn_acc", "rf_acc"],
                "allowed_hue": ["species"],
                "allowed_col": ["scope"]
            },
            "GroupedBarplot": {
                "allowed_col": ["scope", "label"],
                "allowed_row": ["scope", "label", "frac"],
                "allowed_hue": ["scope", "label"],
                "allowed_agg": ["knn_acc", "rf_acc"],
                "allowed_weight": ["initial_row_count", "initial_col_count"]
            }
        }
    }
}

PLOTS = {
    "Barplot": {
        "GroupedBarplot": {
            "controls": {
                "col_facet": True,
                "row_facet": True,
                "hue": {"enabled": True},
                "weighting": {"enabled": True},
                "agg_func": ["count", "mean", "std", "sum"]
            }
        },
        "Histogram": {
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

        plot_type_selection = st.selectbox("Plot Type", list(PLOTS.keys()))
        plot_type = next((k for k in PLOTS.keys() if k.lower() == plot_type_selection.lower()), plot_type_selection)

        subplot_type_selection = st.selectbox("Subplot Type", list(PLOTS[plot_type].keys()))
        subplot_type = next((k for k in PLOTS[plot_type].keys() if k.lower() == subplot_type_selection.lower()), subplot_type_selection)

        controls = PLOTS[plot_type][subplot_type]["controls"]

        db_key = self.shared_state.get("db")
        table_name = self.shared_state.get("table")
        plot_key = subplot_type.lower()

        dataset_cfg = DATA_SPECIFIC_CONFIG.get(db_key, {}).get(table_name, {})
        category_order = dataset_cfg.get("category_order", {})

        data_cfg = next(
            (v for k, v in dataset_cfg.items() if k.lower() == plot_key),
            {}
        )

        all_cols = list(self.df.columns)
        picked_cols = set()

        def case_insensitive_filter(cols, allowed):
            return [c for c in cols if allowed is None or c.lower() in [a.lower() for a in allowed]]

        allowed_row_cols = case_insensitive_filter(all_cols, data_cfg.get("allowed_row"))
        allowed_hue_cols = case_insensitive_filter(all_cols, data_cfg.get("allowed_hue"))
        allowed_col_cols = case_insensitive_filter(all_cols, data_cfg.get("allowed_col"))
        allowed_weight_cols = case_insensitive_filter(all_cols, data_cfg.get("allowed_weight"))
        numeric_cols = list(self.df.select_dtypes(include="number").columns)

        col_facet = None
        row_facet = []
        hue_col = None
        agg_func = "count"
        agg_col = None
        weight_col = None
        weighting_mode = "none"
        max_bins = 10
        max_col_bins = 5

        if controls.get("col_facet"):
            available_cols = [c for c in allowed_col_cols if c not in picked_cols]
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                col_facet = st.selectbox("Col Facet", available_cols)
            if col_facet:
                picked_cols.add(col_facet)
            with col2:
                max_col_bins = st.number_input("Max Col Bins", min_value=2, max_value=20, value=5)

        if controls.get("row_facet"):
            available_cols = [c for c in allowed_row_cols if c not in picked_cols]
            row_facet = st.multiselect("Row Facet(s)", available_cols)
            picked_cols.update(row_facet)

        if controls.get("hue", {}).get("enabled"):
            available_cols = [c for c in allowed_hue_cols if c not in picked_cols]
            hue_col = st.selectbox("Hue Column", available_cols)
            if hue_col:
                picked_cols.add(hue_col)

        if isinstance(controls.get("agg_func"), list):
            agg_func = st.selectbox("Aggregation function", controls["agg_func"])
            available_cols = [c for c in numeric_cols if c not in picked_cols]
            agg_col = st.selectbox("Aggregation column", available_cols)
            st.caption("Note: Aggregation column ignored for count but shown for clarity.")
            if agg_col:
                picked_cols.add(agg_col)

        if controls.get("weighting", {}).get("enabled"):
            weighting_mode = st.selectbox("Weighting mode", ["none", "combined", "additional", "weighted only"])
            if weighting_mode != "none":
                available_cols = [c for c in allowed_weight_cols if c not in picked_cols]
                weight_col = st.selectbox("Weight Column", available_cols)
                if weight_col:
                    picked_cols.add(weight_col)

        fast_mode = st.checkbox("⚡ Fast mode (sample 100 rows)")
        multi_plot = st.checkbox("📄 Multi-plot (single grid figure)", value=True)

        if st.button("Render Plot"):
            self.handle_plot(
                plot_type, subplot_type,
                col_facet, row_facet, hue_col,
                max_bins, fast_mode, multi_plot,
                max_col_bins, agg_func, agg_col,
                category_order, weighting_mode, weight_col
            )

    def handle_plot(self, plot_type, subplot_type,
                    col_facet, row_facet, hue_col,
                    max_bins, fast_mode, multi_plot,
                    max_col_bins, agg_func, agg_col,
                    category_order, weighting_mode, weight_col):
        df_prep = self.prepare_data(
            col_facet, row_facet, hue_col, fast_mode, max_col_bins, agg_col, weight_col
        )

        kwargs = dict(
            col_facet=col_facet,
            row_facet=row_facet,
            hue_col=hue_col,
            max_bins=max_bins,
            multi_plot=multi_plot,
            agg_func=agg_func,
            agg_col=agg_col,
            category_order=category_order,
            weighting_mode=weighting_mode,
            weight_col=weight_col
        )

        init_kwargs = {
            'df': df_prep,
            'col_facet': kwargs['col_facet'],
            'row_facet': kwargs['row_facet'],
            'hue_col': kwargs['hue_col'],
            'category_order': kwargs['category_order']
        }

        # Plot kwargs vary by type:
        if subplot_type == "GroupedBarplot":
            from utils.pages.visu.plots.plot_barplot import GroupedBarplot
            plot_instance = GroupedBarplot(**init_kwargs)
            plot_kwargs = {
                'agg_func': kwargs['agg_func'],
                'agg_col': kwargs['agg_col'],
                'weighting_mode': kwargs['weighting_mode'],
                'weight_col': kwargs['weight_col'],
                'max_bins': kwargs['max_bins'],
                'multi_plot': kwargs['multi_plot']
            }
            result = plot_instance.plot(**plot_kwargs)

        elif subplot_type == "Histogram":
            from utils.pages.visu.plots.plot_barplot import HistogramBarplot
            plot_instance = HistogramBarplot(**init_kwargs)
            plot_kwargs = {
                'max_bins': kwargs['max_bins'],
                'multi_plot': kwargs['multi_plot']
            }
            result = plot_instance.plot(**plot_kwargs)




        if isinstance(result, tuple):
            figs, num_cols = result
        else:
            figs = result
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
        elif figs:
            st.pyplot(figs)

    def prepare_data(self, col_facet, row_facet, hue_col, fast_mode, max_col_bins, agg_col=None, weight_col=None):
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
        if agg_col:
            cols_needed.add(agg_col)
        if weight_col:
            cols_needed.add(weight_col)
        logger.debug(f"Preparing data with columns needed: {cols_needed}")
        df = df[list(cols_needed)].copy()
        if col_facet:
            df = self.reduce_col_facet(df, col_facet, max_col_bins)
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
