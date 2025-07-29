import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from utils.pages.visu.plots.plot_barplot import BarplotBase




class SimpleLineplot(BarplotBase):
    def plot(self, x_col=None, y_col=None, multi_plot=True):
        if not x_col:
            logging.warning("SimpleLineplot.plot: Need x_col.")
            return None

        # row_facet defines y-columns
        y_cols = self.row_facet if self.row_facet else ([y_col] if y_col else [])
        if not y_cols:
            logging.warning("SimpleLineplot.plot: No y columns selected.")
            return None

        col_vals = self._get_col_vals()
        num_rows, num_cols = len(y_cols), len(col_vals)

        if multi_plot:
            fig, axes = plt.subplots(num_rows, num_cols,
                                     figsize=(5*num_cols, 4*num_rows),
                                     squeeze=False)

            for i, y in enumerate(y_cols):
                for j, cval in enumerate(col_vals):
                    ax = axes[i][j]
                    subset = self.df.copy()
                    if cval is not None:
                        subset = subset[subset[self.col_facet] == cval]

                    if subset.empty:
                        ax.set_visible(False)
                        continue

                    sns.lineplot(data=subset, x=x_col, y=y,
                                 hue=self.hue_col, ax=ax)

                    title = f"{y}"
                    if cval is not None:
                        title += f" | {self.col_facet}={cval}"
                    self._set_title_and_subtitle(ax, title)

                    if self.hue_col:
                        handles, labels = ax.get_legend_handles_labels()
                        if handles:
                            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                            ax.legend(handles, labels,
                                    bbox_to_anchor=(1.05, 1), loc="upper left",
                                    borderaxespad=0., fontsize="small")


            plt.tight_layout()
            return fig

        else:
            figs = []
            for i, y in enumerate(y_cols):
                for j, cval in enumerate(col_vals):
                    subset = self.df.copy()
                    if cval is not None:
                        subset = subset[subset[self.col_facet] == cval]
                    if subset.empty:
                        continue

                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.lineplot(data=subset, x=x_col, y=y,
                                 hue=self.hue_col, ax=ax)

                    title = f"{y}"
                    if cval is not None:
                        title += f" | {self.col_facet}={cval}"
                    self._set_title_and_subtitle(ax, title)

                    if self.hue_col:
                        handles, labels = ax.get_legend_handles_labels()
                        if handles:
                            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                            ax.legend(handles, labels,
                                    bbox_to_anchor=(1.05, 1), loc="upper left",
                                    borderaxespad=0., fontsize="small")


                    figs.append(fig)

            return figs, num_cols


