import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
            "layout": self.initial_config.get("layout", "grid"),
            "split_col": self.initial_config.get("split_col", df.columns[0]),
            "hue": self.initial_config.get("hue", None),
            "agg": self.initial_config.get("agg", "count"),
            "vars": self.initial_config.get("vars", []),
            "max_cols": int(self.initial_config.get("max_cols", 5)),
            "max_hue": int(self.initial_config.get("max_hue", 6)),
            "bins": int(self.initial_config.get("bins", 20)),
            "fig_w": float(self.initial_config.get("fig_w", 4)),
            "fig_h": float(self.initial_config.get("fig_h", 3)),
            "palette": self.initial_config.get("palette", "tab10"),
            "mpl_style": self.initial_config.get("mpl_style", "default"),
            "live_preview": bool(self.initial_config.get("live_preview", True)),
        }

    def get_state(self):
        """Expose widget state so it ends up encoded in the share‑URL."""
        return self._state

    def render_controls(self, numeric_cols, categorical_cols):
        """Render sidebar/expander controls and persist state."""
        with st.expander("⚙️ Controls", expanded=True):
            plot_type = st.selectbox(
                "Plot type",
                ["histogram", "bar", "line"],
                index=["histogram", "bar", "line"].index(self._state["plot_type"]),
                help="Select the type of plot to display",
            )

            layout = st.radio(
                "Layout",
                ["grid", "single"],
                index=["grid", "single"].index(self._state["layout"]),
                horizontal=True,
            )

            split_col = st.selectbox(
                "Facet / Split Column",
                options=self.df.columns,
                index=(
                    self.df.columns.get_loc(self._state["split_col"])
                    if self._state["split_col"] in self.df.columns
                    else 0
                ),
            )

            hue_col = None
            if plot_type in {"bar", "line"}:
                hue_col = st.selectbox(
                    "Hue (categorical)",
                    ["None"] + categorical_cols,
                    index=(
                        (["None"] + categorical_cols).index(
                            self._state.get("hue", "None")
                        )
                        if self._state.get("hue") in categorical_cols
                        else 0
                    ),
                    help="Colour grouping",
                )
                hue_col = None if hue_col == "None" else hue_col

            agg_method = st.selectbox(
                "Aggregation",
                ["count", "mean", "median", "std", "min", "max"],
                index=["count", "mean", "median", "std", "min", "max"].index(
                    self._state["agg"]
                ),
            )

            selected_vars = st.multiselect(
                "Variables (numeric)",
                numeric_cols,
                default=[c for c in self._state["vars"] if c in numeric_cols]
                or numeric_cols[:1],
            )

            max_cols = st.number_input(
                "Max facet columns", 1, 20, value=self._state["max_cols"], step=1
            )

            max_hue = None
            if plot_type in {"bar", "line"} and hue_col is not None:
                max_hue = st.number_input(
                    "Max hue categories", 1, 15, value=self._state["max_hue"], step=1
                )

            bins = None
            if plot_type == "histogram":
                bins = st.slider("Bins", 5, 100, value=self._state["bins"], step=1)

            fig_w = st.number_input(
                "Fig width", 2.0, 20.0, value=self._state["fig_w"], step=0.5
            )
            fig_h = st.number_input(
                "Fig height", 2.0, 20.0, value=self._state["fig_h"], step=0.5
            )

            palette = st.selectbox(
                "Colour palette",
                sns.palettes.SEABORN_PALETTES.keys(),
                index=(
                    list(sns.palettes.SEABORN_PALETTES.keys()).index(
                        self._state["palette"]
                    )
                    if self._state["palette"] in sns.palettes.SEABORN_PALETTES
                    else 0
                ),
            )
            mpl_style = st.selectbox(
                "Matplotlib style",
                plt.style.available,
                index=(
                    plt.style.available.index(self._state["mpl_style"])
                    if self._state["mpl_style"] in plt.style.available
                    else 0
                ),
            )

            live_preview = st.checkbox(
                "Live preview", value=self._state.get("live_preview", True)
            )

        self._state.update(
            {
                "plot_type": plot_type,
                "layout": layout,
                "split_col": split_col,
                "hue": hue_col,
                "agg": agg_method,
                "vars": selected_vars,
                "max_cols": int(max_cols),
                "max_hue": (
                    int(max_hue) if max_hue is not None else self._state["max_hue"]
                ),
                "bins": int(bins) if bins is not None else self._state["bins"],
                "fig_w": fig_w,
                "fig_h": fig_h,
                "palette": palette,
                "mpl_style": mpl_style,
                "live_preview": live_preview,
            }
        )

    @st.cache_data(show_spinner=False)
    def aggregate_data(
        self, mask, metric, agg_method, *, bins=None, hue_col=None, categories=None
    ):
        if hue_col is None:
            values = self.df.loc[mask, metric].dropna()
            if bins is None:
                return {"values": values}
            if agg_method == "count":
                counts, bin_edges = np.histogram(values, bins=bins)
                return {
                    "bin_centers": (bin_edges[:-1] + bin_edges[1:]) / 2,
                    "heights": counts,
                    "widths": np.diff(bin_edges),
                }

            counts, bin_edges = np.histogram(values, bins=bins)
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
                    elif agg_method == "min":
                        agg_values[i] = in_bin.min()
                    elif agg_method == "max":
                        agg_values[i] = in_bin.max()
            return {
                "bin_centers": bin_centers,
                "heights": agg_values,
                "widths": np.diff(bin_edges),
            }

        df_subset = self.df.loc[mask, [hue_col, metric]].dropna()
        if df_subset.empty:
            return {"categories": [], "heights": []}
        grouped = df_subset.groupby(hue_col)[metric]
        if agg_method == "count":
            heights = grouped.count()
        elif agg_method == "mean":
            heights = grouped.mean()
        elif agg_method == "median":
            heights = grouped.median()
        elif agg_method == "std":
            heights = grouped.std(ddof=0)
        elif agg_method == "min":
            heights = grouped.min()
        elif agg_method == "max":
            heights = grouped.max()
        else:
            heights = pd.Series(dtype=float)

        cats = [h for h in (categories or []) if h in heights.index]
        for h in heights.index:
            if h not in cats and len(cats) < self._state["max_hue"]:
                cats.append(h)
        return {"categories": cats, "heights": [heights.get(c, 0) for c in cats]}

    # ──────────────────────────────────────────────────────────────────────────
    # Main render routine
    # ──────────────────────────────────────────────────────────────────────────
    def render(self):
        st.subheader("📊 Enhanced Quick Distribution Plotter")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in self.df.columns if c not in numeric_cols]

        if not numeric_cols:
            st.error("No numeric columns available for visualisation.")
            return

        self.render_controls(numeric_cols, categorical_cols)

        if not self._state["vars"]:
            st.warning("Please select at least one numeric variable.")
            return

        plt.style.use(self._state["mpl_style"])
        sns.set_palette(self._state["palette"])

        facet_series = self._prepare_facet_series(
            self._state["split_col"], self._state["max_cols"]
        )
        facet_values = (
            facet_series.dropna().unique().tolist()[: int(self._state["max_cols"])]
        )
        hue_series_full, top_hues = self._prepare_hue(
            self._state["plot_type"], self._state["hue"]
        )

        if self._state["layout"] == "grid":
            fig = self.render_grid_layout(
                facet_series, facet_values, hue_series_full, top_hues
            )
        else:
            fig = self.render_individual_layout(
                facet_series, facet_values, hue_series_full, top_hues
            )

        if all(not ax.get_visible() for ax in fig.axes):
            st.warning("⚠️ Selection produced no visible plot.")
        st.pyplot(fig)
        self.download_plot_button(fig, self._state)

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
        if plot_type not in {"bar", "line"} or hue_col is None:
            return None, []
        hue_series_full = self.df[hue_col].astype(str)
        top_hues = hue_series_full.value_counts().index.tolist()[
            : int(self._state["max_hue"])
        ]
        return hue_series_full, top_hues

    # ──────────────────────────────────────────────────────────────────────────
    # Grid & single renderers
    # ──────────────────────────────────────────────────────────────────────────
    def render_grid_layout(self, facet_series, facet_values, hue_series_full, top_hues):
        """Render a grid of subplots for each variable/facet combination."""
        vars_ = self._state["vars"]
        n_rows, n_cols = len(vars_), len(facet_values)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self._state["fig_w"] * n_cols, self._state["fig_h"] * n_rows),
            squeeze=False,
        )

        for r, metric in enumerate(vars_):
            for c, facet_val in enumerate(facet_values):
                ax = axes[r][c]
                facet_mask = facet_series == facet_val
                self.render_single_plot(
                    ax, facet_mask, metric, hue_series_full, top_hues
                )
                if c == 0:
                    ylabel = (
                        "count" if self._state["agg"] == "count" else self._state["agg"]
                    )
                    ax.set_ylabel(ylabel)
                ax.set_title(f"{metric}\n{self._state['split_col']} = {facet_val}")

        plt.tight_layout()
        return fig

    def render_individual_layout(
        self, facet_series, facet_values, hue_series_full, top_hues
    ):
        metric = self._state["vars"][0]
        facet_val = facet_values[0] if facet_values else "all"
        fig, ax = plt.subplots(figsize=(self._state["fig_w"], self._state["fig_h"]))
        facet_mask = facet_series == facet_val if facet_values else slice(None)

        self.render_single_plot(ax, facet_mask, metric, hue_series_full, top_hues)
        ax.set_title(f"{metric} – {self._state['split_col']} = {facet_val}")
        ylabel = "count" if self._state["agg"] == "count" else self._state["agg"]
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        return fig

    def render_single_plot(self, ax, mask, metric, hue_series_full, top_hues):
        plot_type = self._state["plot_type"]
        if plot_type == "histogram":
            data = self.aggregate_data(
                mask, metric, self._state["agg"], bins=self._state["bins"]
            )
            self._draw_histogram(ax, data)
        elif plot_type == "bar":
            data = self.aggregate_data(
                mask,
                metric,
                self._state["agg"],
                hue_col=self._state["hue"],
                categories=top_hues,
            )
            self._draw_barplot(ax, data, self._state["hue"])
        else:
            if self._state["hue"] is None:
                st.warning("Line plot requires a hue column")
                ax.set_visible(False)
                return
            data = self.aggregate_data(
                mask,
                metric,
                self._state["agg"],
                hue_col=self._state["hue"],
                categories=top_hues,
            )
            self._draw_lineplot(ax, data, self._state["hue"])

    # ──────────────────────────────────────────────────────────────────────────
    # Plot helpers
    # ─────────────────────────────────────────────────────────────────────────-
    def _draw_histogram(self, ax, data):
        if not data:
            ax.set_visible(False)
            return
        if "values" in data:
            ax.hist(
                data["values"],
                bins=self._state["bins"],
                color=sns.color_palette(self._state["palette"])[0],
            )
        else:
            ax.bar(
                data["bin_centers"],
                data["heights"],
                width=data["widths"],
                align="center",
                color=sns.color_palette(self._state["palette"])[0],
            )

    def _draw_barplot(self, ax, data, hue_col):
        categories = data.get("categories", [])
        heights = data.get("heights", [])
        if not categories:
            ax.set_visible(False)
            return

        palette = sns.color_palette(self._state["palette"], len(categories))
        x = np.arange(len(categories))
        bar_width = 0.8
        for idx, (cat, height) in enumerate(zip(categories, heights)):
            ax.bar(idx, height, width=bar_width, label=cat, color=palette[idx])

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        if hue_col is not None:
            ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    def _draw_lineplot(self, ax, data, hue_col):
        categories = data.get("categories", [])
        heights = data.get("heights", [])
        if not categories:
            ax.set_visible(False)
            return

        palette = sns.color_palette(self._state["palette"], len(categories))
        ax.plot(categories, heights, marker="o", color=palette[0])
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right")

    def download_plot_button(self, fig, plot_info):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        vars_tag = "-".join(plot_info.get("vars", [])[:3])
        filename = f"{plot_info.get('plot_type')}_{plot_info.get('agg')}_{plot_info.get('split_col')}_{vars_tag}.png"
        st.download_button(
            "💾 Download PNG", buf.getvalue(), file_name=filename, mime="image/png"
        )
