import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.utils.pages.visu.BaseVisu import BaseVisu


class HistogramVisu(BaseVisu):
    """Quick‑and‑dirty *all‑in‑one* visualiser.

    Supports two families and a toggle to switch between a full **subplot grid**
    or a **single plot** (useful for a rapid peek at just one slice).

    Features
    --------
    * **Histogram** and **Grouped Bar** plots (as before).
    * **Layout toggle** – *"grid"* (rows × columns) **vs** *"single"* (first
      variable & first facet in one axis).
    * **Download button** – exports the current matplotlib figure as a PNG with
      a descriptive filename.
    * Still keeps URL‑round‑trip state so deep‑links keep working.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Init / state round‑trip
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, df, shared_state, initial_config=None):
        super().__init__(df, shared_state, initial_config)

        self._state = {
            "plot_type": self.initial_config.get("plot_type", "histogram"),
            "layout": self.initial_config.get("layout", "grid"),  # new toggle
            "split_col": self.initial_config.get("split_col", df.columns[0]),
            "hue": self.initial_config.get("hue", None),
            "agg": self.initial_config.get("agg", "count"),
            "vars": self.initial_config.get("vars", []),
            "max_cols": int(self.initial_config.get("max_cols", 5)),
            "max_hue": int(self.initial_config.get("max_hue", 6)),
            "bins": int(self.initial_config.get("bins", 20)),
        }

    def get_state(self):
        """Expose widget state so it ends up encoded in the share‑URL."""
        return self._state

    # ──────────────────────────────────────────────────────────────────────────
    # Main render routine
    # ──────────────────────────────────────────────────────────────────────────
    def render(self):
        st.subheader("📊 Quick Distribution Plotter")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in self.df.columns if c not in numeric_cols]

        if not numeric_cols:
            st.error("No numeric columns available for visualisation.")
            return

        # ──────────────────────────────────────────────────────────────────
        # Controls
        # ──────────────────────────────────────────────────────────────────
        with st.expander("⚙️ Controls", expanded=True):
            # Plot family (only two for now)
            plot_type = st.selectbox(
                "Plot type", ["histogram", "bar"],
                index=["histogram", "bar"].index(self._state["plot_type"])
            )

            # Layout toggle ------------------------------------------------
            layout = st.radio(
                "Layout", ["grid", "single"],
                index=["grid", "single"].index(self._state["layout"]),
                horizontal=True,
            )

            split_col = st.selectbox(
                "Facet / Split Column", options=self.df.columns,
                index=self.df.columns.get_loc(self._state["split_col"]) if self._state["split_col"] in self.df.columns else 0,
            )

            hue_col = None
            if plot_type == "bar":
                hue_col = st.selectbox(
                    "Hue (categorical)", ["None"] + categorical_cols,
                    index=(["None"] + categorical_cols).index(self._state["hue"]) if self._state["hue"] in categorical_cols else 0,
                )
                hue_col = None if hue_col == "None" else hue_col

            agg_method = st.selectbox(
                "Aggregation", ["count", "mean", "median", "std"],
                index=["count", "mean", "median", "std"].index(self._state["agg"])
            )

            selected_vars = st.multiselect(
                "Variables (numeric)", numeric_cols,
                default=[c for c in self._state["vars"] if c in numeric_cols] or numeric_cols[:1]
            )

            max_cols = st.number_input(
                "Max facet columns", 1, 20, value=self._state["max_cols"], step=1
            )

            max_hue = None
            if plot_type == "bar" and hue_col is not None:
                max_hue = st.number_input(
                    "Max hue categories", 1, 15, value=self._state["max_hue"], step=1
                )

            bins = None
            if plot_type == "histogram":
                bins = st.slider("Bins", 5, 100, value=self._state["bins"], step=1)

        # Persist state ---------------------------------------------------
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

        if not selected_vars:
            st.warning("Please select at least one numeric variable.")
            return

        # ──────────────────────────────────────────────────────────────────
        # Prepare facet & hue series
        # ──────────────────────────────────────────────────────────────────
        facet_series = self._prepare_facet_series(split_col, max_cols)
        facet_values = facet_series.dropna().unique().tolist()[: int(max_cols)]

        hue_series_full, top_hues = self._prepare_hue(plot_type, hue_col)

        # ──────────────────────────────────────────────────────────────────
        # Plot dispatch
        # ──────────────────────────────────────────────────────────────────
        if layout == "grid":
            fig = self._render_grid(selected_vars, plot_type, agg_method,
                                    facet_series, facet_values,
                                    hue_col, hue_series_full, top_hues)
        else:  # single
            fig = self._render_single(selected_vars, plot_type, agg_method,
                                      facet_series, facet_values,
                                      hue_col, hue_series_full, top_hues)

        st.pyplot(fig)

        # ──────────────────────────────────────────────────────────────────
        # Download button
        # ──────────────────────────────────────────────────────────────────
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        vars_tag = "-".join(selected_vars[:3]) + ("-etc" if len(selected_vars) > 3 else "")
        filename = f"{plot_type}_{agg_method}_{split_col}_{vars_tag}.png"
        st.download_button("💾 Download PNG", buf.getvalue(), file_name=filename, mime="image/png")

    # ──────────────────────────────────────────────────────────────────────────
    # Facet & hue helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_facet_series(self, split_col, max_cols):
        if pd.api.types.is_numeric_dtype(self.df[split_col]):
            return pd.qcut(
                self.df[split_col],
                q=min(int(max_cols), self.df[split_col].nunique()),
                duplicates="drop",
            ).astype(str)
        return self.df[split_col].astype(str)

    def _prepare_hue(self, plot_type, hue_col):
        if plot_type != "bar" or hue_col is None:
            return None, []
        hue_series_full = self.df[hue_col].astype(str)
        top_hues = hue_series_full.value_counts().index.tolist()[: int(self._state["max_hue"])]
        return hue_series_full, top_hues

    # ──────────────────────────────────────────────────────────────────────────
    # Grid & single renderers
    # ──────────────────────────────────────────────────────────────────────────
    def _render_grid(self, selected_vars, plot_type, agg_method,
                     facet_series, facet_values,
                     hue_col, hue_series_full, top_hues):
        n_rows, n_cols = len(selected_vars), len(facet_values)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

        for r, metric in enumerate(selected_vars):
            for c, facet_val in enumerate(facet_values):
                ax = axes[r][c]
                facet_mask = facet_series == facet_val

                if plot_type == "histogram":
                    self._draw_histogram(ax, facet_mask, metric, agg_method)
                else:
                    self._draw_barplot(ax, facet_mask, metric, agg_method,
                                       hue_col=hue_col,
                                       hue_series_full=hue_series_full,
                                       hue_categories=top_hues)

                if c == 0:
                    ylabel = "count" if agg_method == "count" else agg_method
                    ax.set_ylabel(ylabel)
                ax.set_title(f"{metric}\n{self._state['split_col']} = {facet_val}")

        plt.tight_layout()
        return fig

    def _render_single(self, selected_vars, plot_type, agg_method,
                       facet_series, facet_values,
                       hue_col, hue_series_full, top_hues):
        metric = selected_vars[0]
        facet_val = facet_values[0] if facet_values else "all"
        fig, ax = plt.subplots(figsize=(6, 4))
        facet_mask = facet_series == facet_val if facet_values else slice(None)

        if plot_type == "histogram":
            self._draw_histogram(ax, facet_mask, metric, agg_method)
        else:
            self._draw_barplot(ax, facet_mask, metric, agg_method,
                               hue_col=hue_col,
                               hue_series_full=hue_series_full,
                               hue_categories=top_hues)

        ax.set_title(f"{metric} – {self._state['split_col']} = {facet_val}")
        ylabel = "count" if agg_method == "count" else agg_method
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    # Plot helpers (unchanged logic)
    # ─────────────────────────────────────────────────────────────────────────-
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
                    agg_values[i] = in_bin.mean()
                elif agg_method == "median":
                    agg_values[i] = in_bin.median()
                elif agg_method == "std":
                    agg_values[i] = in_bin.std(ddof=0)
        ax.bar(bin_centers, agg_values, width=np.diff(bin_edges), align="center")

    def _draw_barplot(self, ax, mask, metric, agg_method, *, hue_col, hue_series_full, hue_categories):
        if hue_col is None:
            subset = self.df.loc[mask, metric].dropna()
            if subset.empty:
                ax.set_visible(False)
                return

            height = len(subset) if agg_method == "count" else getattr(subset, agg_method)()
            ax.bar(0, height, width=0.6, color="tab:blue")
            ax.set_xticks([0])
            ax.set_xticklabels(["all"], rotation=0)
            return

        df_subset = self.df.loc[mask, [hue_col, metric]].dropna()
        if df_subset.empty:
            ax.set_visible(False)
            return

        grouped = df_subset.groupby(hue_col)[metric]
        if agg_method == "count":
            heights = grouped.count()
        elif agg_method == "mean":
            heights = grouped.mean()
        elif agg_method == "median":
            heights = grouped.median()
        elif agg_method == "std":
            heights = grouped.std(ddof=0)
        else:
            heights = pd.Series(dtype=float)

        categories = [h for h in hue_categories if h in heights.index]
        for h in heights.index:
            if h not in categories and len(categories) < self._state["max_hue"]:
                categories.append(h)

        x = np.arange(len(categories))
        bar_width = 0.8
        for idx, cat in enumerate(categories):
            height = heights.get(cat, 0)
            ax.bar(idx, height, width=bar_width, label=cat)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        if ax.get_subplotspec().is_first_col() and ax.get_subplotspec().is_first_row() and hue_col is not None:
            ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc="upper left")
