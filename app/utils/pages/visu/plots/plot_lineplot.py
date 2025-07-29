import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from utils.pages.visu.plots.plot_barplot import BarplotBase

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)

class LineSumPlot(BarplotBase):
    """
    High-performance LineSumPlot:
    - Fast using sns.ecdfplot internally
    - Consistent styling with GroupedBarplot
    """

    def plot(self, agg_col=None, weighting_mode="none", weight_col=None, multi_plot=True):
        logger.info(f"LineSumPlot.plot: row_facet={self.row_facet}, agg_col={agg_col}")

        if not self.row_facet or len(self.row_facet) != 1:
            logger.warning("LineSumPlot.plot: Requires exactly one row_facet column.")
            return None

        row_facet_col = self.row_facet[0]
        if self.category_order and row_facet_col in self.category_order:
            row_vals = [v for v in self.category_order[row_facet_col] if v in self.df[row_facet_col].unique()]
        else:
            row_vals = sorted(self.df[row_facet_col].dropna().unique())

        col_vals = self._get_col_vals()
        num_rows = len(row_vals)
        num_cols = len(col_vals)

        agg_cols = agg_col if isinstance(agg_col, list) else [agg_col]
        if not agg_cols or (len(agg_cols) == 1 and agg_cols[0] is None):
            logger.warning("LineSumPlot.plot: No valid agg_cols selected.")
            return None

        if multi_plot:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
            axes = np.atleast_2d(axes)

            for i, r_val in enumerate(row_vals):
                for j, cval in enumerate(col_vals):
                    ax = axes[i][j]
                    subset = self.df[self.df[row_facet_col] == r_val]
                    if cval is not None:
                        subset = subset[subset[self.col_facet] == cval]
                    if subset.empty:
                        ax.set_visible(False)
                        continue

                    for metric in agg_cols:
                        if metric not in subset.columns:
                            logger.warning(f"Skipping missing metric column: {metric}")
                            continue

                        if self.hue_col and self.hue_col in subset.columns:
                            sns.ecdfplot(
                                data=subset, x=metric, hue=self.hue_col, ax=ax,
                                stat="proportion", complementary=False,
                                weights=subset[weight_col] if weighting_mode in ["combined", "weighted only"] and weight_col in subset.columns else None
                            )
                        else:
                            sns.ecdfplot(
                                data=subset, x=metric, ax=ax,
                                stat="proportion", complementary=False,
                                weights=subset[weight_col] if weighting_mode in ["combined", "weighted only"] and weight_col in subset.columns else None,
                                label=metric
                            )

                    title = f"{row_facet_col}={r_val}"
                    if cval is not None:
                        title += f" | {self.col_facet}={cval}"

                    self._apply_common_style(ax, ylabel="Cumulative fraction", max_y=1.0)
                    self._set_title_and_subtitle(ax, title, "")

                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

            plt.tight_layout()
            return fig

        else:
            figs = []
            for i, r_val in enumerate(row_vals):
                for j, cval in enumerate(col_vals):
                    subset = self.df[self.df[row_facet_col] == r_val]
                    if cval is not None:
                        subset = subset[subset[self.col_facet] == cval]
                    if subset.empty:
                        continue

                    fig, ax = plt.subplots(figsize=(6, 4))

                    for metric in agg_cols:
                        if metric not in subset.columns:
                            logger.warning(f"Skipping missing metric column: {metric}")
                            continue

                        if self.hue_col and self.hue_col in subset.columns:
                            sns.ecdfplot(
                                data=subset, x=metric, hue=self.hue_col, ax=ax,
                                stat="proportion", complementary=False,
                                weights=subset[weight_col] if weighting_mode in ["combined", "weighted only"] and weight_col in subset.columns else None
                            )
                        else:
                            sns.ecdfplot(
                                data=subset, x=metric, ax=ax,
                                stat="proportion", complementary=False,
                                weights=subset[weight_col] if weighting_mode in ["combined", "weighted only"] and weight_col in subset.columns else None,
                                label=metric
                            )

                    title = f"{row_facet_col}={r_val}"
                    if cval is not None:
                        title += f" | {self.col_facet}={cval}"

                    self._apply_common_style(ax, ylabel="Cumulative fraction", max_y=1.0)
                    self._set_title_and_subtitle(ax, title, "")

                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

                    figs.append(fig)

            return figs, num_cols
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DensityPlot(BarplotBase):
    """
    Histogram-as-lineplot implementation with hue support and filled areas under curves.
    Efficient handling, no removal of artists for multiplot mode.
    """

    def plot(self, agg_cols=None, weighting_mode="none", weight_col=None, multi_plot=True):
        logger.info(f"DensityPlot.plot agg_cols={agg_cols} weighting_mode={weighting_mode} weight_col={weight_col}")

        if agg_cols is None:
            agg_cols = self.agg_col

        if not isinstance(agg_cols, list):
            agg_cols = [agg_cols]

        if not self.row_facet:
            return None

        row_facet = self.row_facet[0]
        row_vals = self.df[row_facet].unique()
        col_vals = self._get_col_vals()
        num_rows = len(row_vals)
        num_cols = len(col_vals)

        bin_count = 50
        x_range = (0, 1)

        plot_specs = []  # Collect plot specs for later rendering in multi_plot

        for r_val in row_vals:
            for c_val in col_vals:
                subset = self.df[self.df[row_facet] == r_val]
                if c_val is not None:
                    subset = subset[subset[self.col_facet] == c_val]
                if subset.empty:
                    continue
                plot_specs.append((r_val, c_val, subset))

        if multi_plot:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)
            axes = np.atleast_2d(axes)

            for idx, (r_val, c_val, subset) in enumerate(plot_specs):
                i = list(row_vals).index(r_val)
                j = list(col_vals).index(c_val)
                ax = axes[i][j]

                for metric in agg_cols:
                    if metric not in subset.columns:
                        continue
                    
                    if self.hue_col and self.hue_col in subset.columns:
                        for hue_val, hue_df in subset.groupby(self.hue_col):
                            data_vals = hue_df[metric].dropna()
                            weights = None
                            if weighting_mode in ["combined", "weighted only"] and weight_col in hue_df.columns:
                                weights = hue_df.loc[data_vals.index, weight_col]

                            counts, bin_edges = np.histogram(data_vals, bins=bin_count, range=x_range, density=True, weights=weights)
                            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                            ax.plot(bin_centers, counts, label=f"{metric} | {hue_val}")
                            ax.fill_between(bin_centers, counts, alpha=0.3)
                    else:
                        data_vals = subset[metric].dropna()
                        weights = None
                        if weighting_mode in ["combined", "weighted only"] and weight_col in subset.columns:
                            weights = subset.loc[data_vals.index, weight_col]

                        counts, bin_edges = np.histogram(data_vals, bins=bin_count, range=x_range, density=True, weights=weights)
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                        ax.plot(bin_centers, counts, label=metric)
                        ax.fill_between(bin_centers, counts, alpha=0.3)

                title = f"{row_facet}={r_val}"
                if c_val is not None:
                    title += f" | {self.col_facet}={c_val}"

                self._set_title_and_subtitle(ax, title)
                self._apply_common_style(ax, ylabel="Density")

                if self.hue_col or len(agg_cols) > 1:
                    ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

            plt.tight_layout()
            return fig

        else:
            figs = []
            for r_val, c_val, subset in plot_specs:
                fig, ax = plt.subplots(figsize=(6, 4))
                for metric in agg_cols:
                    if metric not in subset.columns:
                        continue

                    if self.hue_col and self.hue_col in subset.columns:
                        for hue_val, hue_df in subset.groupby(self.hue_col):
                            data_vals = hue_df[metric].dropna()
                            weights = None
                            if weighting_mode in ["combined", "weighted only"] and weight_col in hue_df.columns:
                                weights = hue_df.loc[data_vals.index, weight_col]

                            counts, bin_edges = np.histogram(data_vals, bins=bin_count, range=x_range, density=True, weights=weights)
                            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                            ax.plot(bin_centers, counts, label=f"{metric} | {hue_val}")
                            ax.fill_between(bin_centers, counts, alpha=0.3)
                    else:
                        data_vals = subset[metric].dropna()
                        weights = None
                        if weighting_mode in ["combined", "weighted only"] and weight_col in subset.columns:
                            weights = subset.loc[data_vals.index, weight_col]

                        counts, bin_edges = np.histogram(data_vals, bins=bin_count, range=x_range, density=True, weights=weights)
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                        ax.plot(bin_centers, counts, label=metric)
                        ax.fill_between(bin_centers, counts, alpha=0.3)

                title = f"{row_facet}={r_val}"
                if c_val is not None:
                    title += f" | {self.col_facet}={c_val}"

                self._set_title_and_subtitle(ax, title)
                self._apply_common_style(ax, ylabel="Density")

                if self.hue_col or len(agg_cols) > 1:
                    ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

                figs.append(fig)

            return figs, num_cols
