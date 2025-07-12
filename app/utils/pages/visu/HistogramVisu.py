import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.utils.pages.visu.BaseVisu import BaseVisu


class HistogramVisu(BaseVisu):
    """All‑in‑one quick visualiser – now with **line plots** and an
    **individual‑grid** layout that still preserves columns. The file is still
    a single monolith for rapid iteration.

    Layouts
    --------
    * **grid** – one combined matplotlib figure (legacy behaviour)
    * **individual** – renders each facet×variable panel *in a Streamlit column
      grid*, so you can download every panel PNG separately *while keeping the
      visual alignment*.

    Plot types implemented
    ----------------------
    * **histogram** – count / mean / median / std per bin
    * **bar**       – grouped bars with optional hue
    * **line**      – turns the same grouped‑values logic into *lines* (one per
      hue category). Requires a hue column so we have more than a single point.
    """

    # ─────────────────────────── Init / state ────────────────────────────
    def __init__(self, df, shared_state, initial_config=None):
        super().__init__(df, shared_state, initial_config)
        self._state = {
            "plot_type": self.initial_config.get("plot_type", "histogram"),
            "layout": self.initial_config.get("layout", "grid"),  # grid | individual
            "split_col": self.initial_config.get("split_col", df.columns[0]),
            "hue": self.initial_config.get("hue", None),
            "agg": self.initial_config.get("agg", "count"),
            "vars": self.initial_config.get("vars", []),
            "max_cols": int(self.initial_config.get("max_cols", 5)),
            "max_hue": int(self.initial_config.get("max_hue", 6)),
            "bins": int(self.initial_config.get("bins", 20)),
        }

    def get_state(self):
        return self._state

    # ──────────────────────────── Render entry ───────────────────────────
    def render(self):
        st.subheader("📊 Quick Distribution Plotter")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in self.df.columns if c not in numeric_cols]
        if not numeric_cols:
            st.error("No numeric columns available for visualisation.")
            return

        # ───────────────────────── Control panel ─────────────────────────
        with st.expander("⚙️ Controls", expanded=True):
            plot_type = st.selectbox(
                "Plot type", ["histogram", "bar", "line"],
                index=["histogram", "bar", "line"].index(self._state["plot_type"]),
            )

            layout = st.radio(
                "Layout", ["grid", "individual"],
                index=["grid", "individual"].index(self._state["layout"]),
                horizontal=True,
            )

            split_col = st.selectbox(
                "Facet / Split Column", self.df.columns,
                index=self.df.columns.get_loc(self._state["split_col"]) if self._state["split_col"] in self.df.columns else 0,
            )

            # Hue only relevant for bar/line
            hue_col = None
            if plot_type in {"bar", "line"}:
                hue_col = st.selectbox(
                    "Hue (categorical)", ["None"] + categorical_cols,
                    index=(["None"] + categorical_cols).index(self._state["hue"]) if self._state["hue"] in categorical_cols else 0,
                )
                hue_col = None if hue_col == "None" else hue_col

            agg_opts = ["count", "mean", "median", "std"]
            agg_method = st.selectbox(
                "Aggregation", agg_opts,
                index=agg_opts.index(self._state["agg"]) if self._state["agg"] in agg_opts else 0,
            )

            selected_vars = st.multiselect(
                "Variables (numeric)", numeric_cols,
                default=[c for c in self._state["vars"] if c in numeric_cols] or numeric_cols[:1],
            )

            max_cols = st.number_input("Max facet columns", 1, 20, value=self._state["max_cols"], step=1)

            max_hue = None
            if plot_type in {"bar", "line"} and hue_col is not None:
                max_hue = st.number_input("Max hue categories", 1, 15, value=self._state["max_hue"], step=1)

            bins = None
            if plot_type == "histogram":
                bins = st.slider("Bins", 5, 100, value=self._state["bins"], step=1)

        # ───────────────────── Persist widget state ──────────────────────
        self._state.update({
            "plot_type": plot_type,
            "layout": layout,
            "split_col": split_col,
            "hue": hue_col,
            "agg": agg_method,
            "vars": selected_vars,
            "max_cols": int(max_cols),
            "max_hue": int(max_hue) if max_hue is not None else self._state["max_hue"],
            "bins": int(bins) if bins is not None else self._state["bins"],
        })

        # ───────────────────── Basic validation ─────────────────────────
        if not selected_vars:
            st.warning("Please select at least one numeric variable.")
            return
        if plot_type == "line" and hue_col is None:
            st.warning("Line plot needs a hue column so we have multiple lines – please pick one.")
            return

        # ───────────────────── Data prep: facet & hue ────────────────────
        facet_series = self._prepare_facet_series(split_col, max_cols)
        facet_values = facet_series.dropna().unique().tolist()[: int(max_cols)]
        hue_series_full, top_hues = self._prepare_hue(plot_type, hue_col)

        # ───────────────────── Render dispatch ───────────────────────────
        if layout == "grid":
            fig = self._render_grid(selected_vars, plot_type, agg_method,
                                    facet_series, facet_values,
                                    hue_col, hue_series_full, top_hues)
            st.pyplot(fig)
            self._offer_download(fig, plot_type, agg_method, split_col, selected_vars)
        else:  # individual grid layout keeping columns
            self._render_individual_grid(selected_vars, plot_type, agg_method,
                                         facet_series, facet_values,
                                         hue_col, hue_series_full, top_hues)

    # ───────────────────── Download helper ──────────────────────────────
    def _offer_download(self, fig, plot_type, agg_method, split_col, selected_vars, facet_val=None):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        vars_tag = "-".join(selected_vars[:3]) + ("-etc" if len(selected_vars) > 3 else "")
        facet_tag = f"_{facet_val}" if facet_val is not None else ""
        filename = f"{plot_type}_{agg_method}_{split_col}{facet_tag}_{vars_tag}.png"
        st.download_button("💾 Download PNG", buf.getvalue(), file_name=filename, mime="image/png")

    # ───────────────────── Prep helpers ────────────────────────────────
    def _prepare_facet_series(self, split_col, max_cols):
        if pd.api.types.is_numeric_dtype(self.df[split_col]):
            return pd.qcut(self.df[split_col], q=min(int(max_cols), self.df[split_col].nunique()), duplicates="drop").astype(str)
        return self.df[split_col].astype(str)

    def _prepare_hue(self, plot_type, hue_col):
        if plot_type not in {"bar", "line"} or hue_col is None:
            return None, []
        hue_series_full = self.df[hue_col].astype(str)
        top_hues = hue_series_full.value_counts().index.tolist()[: int(self._state["max_hue"])]
        return hue_series_full, top_hues

    # ───────────────────── Grid renderer (one figure) ──────────────────
    def _render_grid(self, selected_vars, plot_type, agg_method,
                     facet_series, facet_values,
                     hue_col, hue_series_full, top_hues):
        n_rows, n_cols = len(selected_vars), len(facet_values)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
        for r, metric in enumerate(selected_vars):
            for c, facet_val in enumerate(facet_values):
                ax = axes[r][c]
                facet_mask = facet_series == facet_val
                self._dispatch_draw(ax, facet_mask, plot_type, metric, agg_method,
                                    hue_col, hue_series_full, top_hues)
                if c == 0:
                    ax.set_ylabel("count" if agg_method == "count" else agg_method)
                ax.set_title(f"{metric}\n{self._state['split_col']} = {facet_val}")
        plt.tight_layout()
        return fig

    # ───────── Individual‑grid renderer (Streamlit columns) ────────────
    def _render_individual_grid(self, selected_vars, plot_type, agg_method,
                                facet_series, facet_values,
                                hue_col, hue_series_full, top_hues):
        n_cols = max(len(facet_values), 1)
        for metric in selected_vars:
            st.markdown(f"#### {metric}")
            cols = st.columns(n_cols)
            for col_idx, facet_val in enumerate(facet_values or ["all"]):
                with cols[col_idx]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    facet_mask = facet_series == facet_val if facet_val != "all" else slice(None)
                    self._dispatch_draw(ax, facet_mask, plot_type, metric, agg_method,
                                        hue_col, hue_series_full, top_hues)
                    ax.set_ylabel("count" if agg_method == "count" else agg_method)
                    ax.set_title(f"{facet_val}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    self._offer_download(fig, plot_type, agg_method, self._state['split_col'], [metric], facet_val)

    # ───────────────────── Draw dispatcher ─────────────────────────────
    def _dispatch_draw(self, ax, mask, plot_type, metric, agg_method,
                       hue_col, hue_series_full, hue_categories):
        if plot_type == "histogram":
            self._draw_histogram(ax, mask, metric, agg_method)
        elif plot_type == "bar":
            self._draw_barplot(ax, mask, metric, agg_method,
                               hue_col=hue_col,
                               hue_series_full=hue_series_full,
                               hue_categories=hue_categories)
        elif plot_type == "line":
            self._draw_lineplot(ax, mask, metric, agg_method,
                                hue_col=hue_col,
                                hue_series_full=hue_series_full,
                                hue_categories=hue_categories)

    # ───────────────────── Primitive drawers ───────────────────────────
    def _draw_histogram(self, ax, mask, metric, agg_method):
        values = self.df.loc[mask, metric].dropna()
        if values.empty:
            ax.set_visible(False)
            return
        if agg_method == "count":
            ax.hist(values, bins=self._state["bins"])
            return
        counts, bin_edges = np.histogram(values, bins=self._state["bins"])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_ids = np.digitize(values, bin_edges) - 1
        agg_values = np.zeros(len(bin_edges) - 1)
        for i in range(len(bin_edges) - 1):
            in_bin = values[bin_ids == i]
            if len(in_bin):
                if agg_method == "mean":
                    agg
