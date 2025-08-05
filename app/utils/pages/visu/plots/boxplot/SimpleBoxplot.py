import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
from app.utils.pages.visu.plots.PlotBase import PlotBase

logger = logging.getLogger("plotbase")

class SimpleBoxplot(PlotBase):
    """
    Boxplot:
    - row_facet = numeric variable(s) to plot
    - col_facet = split into subplot columns
    - hue_col = optional subgroup coloring
    - x_col = optional grouping on x-axis
    """

    def plot(self, multi_plot=True, x_col=None):
        row_vars = self.row_facet if isinstance(self.row_facet, list) else [self.row_facet]
        col_vals = self._get_col_vals()
        num_rows, num_cols = len(row_vars), len(col_vals)

        if multi_plot:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)
            axes = np.atleast_2d(axes)

            for i, var in enumerate(row_vars):
                if var not in self.df.columns:
                    logger.warning(f"Skipping missing column {var}")
                    continue
                for j, cval in enumerate(col_vals):
                    ax = axes[i][j]
                    subset = self.df.copy()
                    if cval is not None:
                        subset = subset[subset[self.col_facet] == cval]
                    if subset.empty:
                        ax.set_visible(False)
                        continue

                    sns.boxplot(
                        data=subset,
                        y=var,
                        x=x_col,
                        hue=self.hue_col,
                        ax=ax
                    )

                    title = f"{var}"
                    if cval is not None:
                        title += f" | {self.col_facet}={cval}"
                    self._set_title_and_subtitle(ax, title)
                    self._apply_common_style(ax, ylabel=var)

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

                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(
                        data=subset,
                        y=var,
                        x=x_col,
                        hue=self.hue_col,
                        ax=ax
                    )

                    title = f"{var}"
                    if cval is not None:
                        title += f" | {self.col_facet}={cval}"
                    self._set_title_and_subtitle(ax, title)
                    self._apply_common_style(ax, ylabel=var)

                    figs.append(fig)
            return figs, num_cols
