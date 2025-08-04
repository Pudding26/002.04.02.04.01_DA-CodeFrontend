# plot_barplot.py (excerpt)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from app.utils.pages.visu.plots.PlotBase import PlotBase


class HistogramBarplot(PlotBase):
    """
    Simple histogram:
    - row_facet defines which numeric columns to histogram
    - col_facet splits into subplot columns
    - hue_col overlays/grouped categories
    - y-axis is always counts
    """

    def plot(self, max_bins=20, multi_plot=True, plot_mode="bars", value_range=None):
        logging.info(f"HistogramBarplot.plot max_bins={max_bins}")

        if not self.row_facet:
            logging.warning("HistogramBarplot: No row_facet variables selected.")
            return None

        # row_facet = list of metrics
        row_vars = self.row_facet if isinstance(self.row_facet, list) else [self.row_facet]
        col_vals = self._get_col_vals()
        num_rows, num_cols = len(row_vars), len(col_vals)



        if multi_plot:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)
            axes = np.atleast_2d(axes)

            for i, var in enumerate(row_vars):
                if var not in self.df.columns:
                    logging.warning(f"Skipping missing column {var}")
                    continue
                for j, cval in enumerate(col_vals):
                    ax = axes[i][j]
                    subset = self.df.copy()
                    if cval is not None:
                        subset = subset[subset[self.col_facet] == cval]

                    if subset.empty:
                        ax.set_visible(False)
                        continue
                    
                    if value_range is not None:
                        ax.set_xlim(value_range[0], value_range[1])

                    if plot_mode in ["bars", "both"]:
                        sns.histplot(
                            data=subset,
                            x=var,
                            hue=self.hue_col,
                            bins=max_bins,
                            ax=ax,
                            element="step",
                            stat="count",
                            common_norm=False
                        )

                    # Add KDE line(s)
                    if plot_mode in ["kde", "both"]:

                        sns.kdeplot(
                            data=subset,
                            fill=True,
                            x=var,
                            hue=self.hue_col,
                            ax=ax,
                            common_norm=False,
                            legend=False  # avoid duplicate legends
                        )

                    title = f"{var}"
                    if cval is not None:
                        title += f" | {self.col_facet}={cval}"
                    self._set_title_and_subtitle(ax, title)
                    self._apply_common_style(ax, ylabel="Count")

            plt.tight_layout()
            return fig

        else:
            figs = []
            for i, var in enumerate(row_vars):
                if var not in self.df.columns:
                    continue
                for j, cval in enumerate(col_vals):
                    subset = self.df.copy()
                    if cval is not None:
                        subset = subset[subset[self.col_facet] == cval]
                    if subset.empty:
                        continue

                    if value_range is not None:
                        ax.set_xlim(value_range[0], value_range[1])

                    fig, ax = plt.subplots(figsize=(6, 4))
                    if plot_mode in ["bars", "both"]:
                        sns.histplot(
                            data=subset,
                            x=var,
                            hue=self.hue_col,
                            bins=max_bins,
                            ax=ax,
                            element="step",
                            stat="count"
                        )
                    if plot_mode in ["kde", "both"]:
                        sns.kdeplot(
                            data=subset,
                            x=var,
                            fill=True,
                            hue=self.hue_col,
                            ax=ax,
                            common_norm=False,
                            legend=False  # avoid duplicate legends
                        )

                    title = f"{var}"
                    if cval is not None:
                        title += f" | {self.col_facet}={cval}"
                    self._set_title_and_subtitle(ax, title)
                    self._apply_common_style(ax, ylabel="Count")

                    figs.append(fig)

            return figs, num_cols
