import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)


class BarplotBase:
    def __init__(self, df, col_facet=None, row_facet=None, hue_col=None, category_order=None):
        self.df = df
        self.col_facet = col_facet
        self.row_facet = row_facet or []
        self.hue_col = hue_col
        self.category_order = category_order or {}

    def _get_col_vals(self):
        return self.df[self.col_facet].unique() if self.col_facet else [None]

    def _get_order(self, key):
        return self.category_order.get(key) if key and self.category_order else None

    def _apply_common_style(self, ax, y_label, max_y=None):
        ax.set_ylabel(y_label)
        ax.tick_params(axis='x', rotation=45)
        if max_y:
            ax.set_ylim(0, max_y)

    def _set_title_and_subtitle(self, ax, title, subtitle=None):
        ax.text(0.5, 1.08, title,
                transform=ax.transAxes,
                fontsize='medium', fontweight='bold',
                ha='center', va='bottom')
        if subtitle:
            ax.text(0.5, 1.05, subtitle,
                    transform=ax.transAxes,
                    fontsize='small', ha='center', va='top')



class GroupedBarplot(BarplotBase):
    """
    Grouped barplot supporting facets, hue, aggregation, weighting modes.

    Weighting modes:
    - "none": Normal behavior.
    - "combined": Weighted overlay shown with transparency.
    - "additional": Weighted version as additional row facet.
    - "weighted only": Only weighted version shown.
    """

    def plot(self, agg_func="count", agg_col=None, weighting_mode="none", weight_col=None,
             max_bins=10, multi_plot=True):
        logger.info(f"GroupedBarplot.plot weighting_mode={weighting_mode}, weight_col={weight_col}")

        if not self.row_facet:
            return None

        col_vals = self._get_col_vals()
        num_rows = len(self.row_facet)
        num_cols = len(col_vals)

        figs = []
        plot_data = []
        y_max_per_row = {}
        effective_row_facet = list(self.row_facet)

        if weighting_mode == "additional":
            effective_row_facet += [f"Weighted: {r}" for r in self.row_facet]

        # Compute plot_data and y_max
        for rfacet in effective_row_facet:
            is_weighted = rfacet.startswith("Weighted: ")
            base_rfacet = rfacet.replace("Weighted: ", "") if is_weighted else rfacet

            if pd.api.types.is_numeric_dtype(self.df[base_rfacet]):
                self.df[base_rfacet] = pd.qcut(self.df[base_rfacet], q=max_bins, duplicates="drop")
                self.df[base_rfacet] = self.df[base_rfacet].apply(lambda x: f"{x.left:.2f}-{x.right:.2f}")

            y_label = f"{agg_func} [{agg_col}]" if agg_func != "count" else f"{agg_func}"
            y_vals = []

            for cval in col_vals:
                subset = self.df
                if cval is not None:
                    subset = subset[subset[self.col_facet] == cval]

                if subset.empty:
                    continue

                group_keys = [base_rfacet]
                if self.hue_col:
                    group_keys.append(self.hue_col)
                group_keys = list(dict.fromkeys(group_keys))

                df_plot = None

                def weighted_agg(subset_df):
                    if agg_func == "count":
                        return subset_df.groupby(group_keys)[weight_col].sum().reset_index(name="value")
                    elif agg_func == "sum":
                        return subset_df.groupby(group_keys).apply(lambda x: np.sum(x[agg_col] * x[weight_col])).reset_index(name="value")
                    elif agg_func == "mean":
                        return subset_df.groupby(group_keys).apply(lambda x: np.average(x[agg_col], weights=x[weight_col])).reset_index(name="value")
                    elif agg_func == "std":
                        return subset_df.groupby(group_keys).apply(lambda x: np.sqrt(np.cov(x[agg_col], aweights=x[weight_col]))).reset_index(name="value")
                    else:
                        raise ValueError(f"Unsupported agg_func: {agg_func}")

                if weighting_mode == "weighted only" or is_weighted:
                    df_plot = weighted_agg(subset)
                else:
                    if agg_func == "count":
                        df_plot = subset.groupby(group_keys).size().reset_index(name="value")
                    else:
                        if agg_col not in subset.columns:
                            continue
                        df_plot = subset.groupby(group_keys)[agg_col].agg(agg_func).reset_index(name="value")

                plot_data.append((rfacet, cval, df_plot))
                y_vals.extend(df_plot["value"].tolist())

            if y_vals:
                y_max_per_row[rfacet] = max(y_vals)

        # Plotting
        if multi_plot:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)
            axes = np.atleast_2d(axes)
            handles, labels = None, None

            for i, rfacet in enumerate(self.row_facet):
                for j, cval in enumerate(col_vals):
                    ax = axes[i][j]
                    match = [item for item in plot_data if item[0] == rfacet and item[1] == cval]
                    if not match:
                        ax.set_visible(False)
                        continue

                    _, _, df_plot = match[0]
                    order = self._get_order(rfacet)
                    hue_order = self._get_order(self.hue_col)

                    sns.barplot(data=df_plot, x=rfacet, y="value", hue=self.hue_col, ax=ax,
                                order=order, hue_order=hue_order)

                    # Combined overlay
                    if weighting_mode == "combined" and weight_col:
                        df_weighted = weighted_agg(subset)
                        sns.barplot(data=df_weighted, x=rfacet, y="value", hue=self.hue_col, ax=ax,
                                    order=order, hue_order=hue_order, alpha=0.4, legend=False)

                    subtitle = None
                    if weighting_mode == "combined" and weight_col:
                        subtitle = f"Weighted overlay using '{weight_col}'"

                    self._apply_common_style(ax, y_label, max_y=y_max_per_row.get(rfacet))
                    
                    main_title = f"{rfacet}" + (f" | {self.col_facet}={cval}" if cval else "")
                    subtitle = None
                    if weighting_mode == "combined" and weight_col:
                        subtitle = f"Weighted overlay shown (alpha=0.4) using '{weight_col}'"

                    self._set_title_and_subtitle(ax, main_title, subtitle)

                    if self.hue_col and handles is None:
                        handles, labels = ax.get_legend_handles_labels()
                    if self.hue_col:
                        ax.get_legend().remove()

            if self.hue_col and handles:
                fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

            plt.tight_layout()
            return fig

        else:
            for rfacet in effective_row_facet:
                for cval in col_vals:
                    match = [item for item in plot_data if item[0] == rfacet and item[1] == cval]
                    if not match:
                        continue

                    _, _, df_plot = match[0]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    order = self._get_order(rfacet)
                    hue_order = self._get_order(self.hue_col)

                    sns.barplot(data=df_plot, x=rfacet.replace("Weighted: ", ""), y="value", hue=self.hue_col, ax=ax,
                                order=order, hue_order=hue_order)

                    if weighting_mode == "combined" and weight_col and not rfacet.startswith("Weighted: "):
                        df_weighted = weighted_agg(subset)
                        sns.barplot(data=df_weighted, x=rfacet, y="value", hue=self.hue_col, ax=ax,
                                    order=order, hue_order=hue_order, alpha=0.4, legend=False)

                    subtitle = None
                    if weighting_mode == "combined" and weight_col:
                        subtitle = f"Weighted overlay using '{weight_col}'"

                    self._apply_common_style(ax, y_label, max_y=y_max_per_row.get(rfacet))



                    main_title = f"{rfacet}" + (f" | {self.col_facet}={cval}" if cval else "")
                    subtitle = None
                    if weighting_mode == "combined" and weight_col:
                        subtitle = f"Weighted overlay shown (alpha=0.4) using '{weight_col}'"

                    self._set_title_and_subtitle(ax, main_title, subtitle)


                    if self.hue_col:
                        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

                    figs.append(fig)

            return figs, num_cols


class HistogramBarplot(BarplotBase):
    """
    Histogram barplot supporting row/col facets, consistent styling.
    """

    def plot(self, max_bins=10, multi_plot=True):
        logger.info(f"HistogramBarplot.plot max_bins={max_bins}")

        if not self.row_facet:
            return None

        col_vals = self._get_col_vals()
        num_rows = len(self.row_facet)
        num_cols = len(col_vals)
        figs = []

        for rfacet in self.row_facet:
            if pd.api.types.is_numeric_dtype(self.df[rfacet]):
                self.df[rfacet] = pd.qcut(self.df[rfacet], q=max_bins, duplicates="drop")
                self.df[rfacet] = self.df[rfacet].apply(lambda x: f"{x.left:.2f}-{x.right:.2f}")

            for cval in col_vals:
                subset = self.df
                if cval is not None:
                    subset = subset[subset[self.col_facet] == cval]

                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(subset[rfacet].dropna(), bins=max_bins, ax=ax)

                title = f"{rfacet}" + (f" | {self.col_facet}={cval}" if cval else "")
                self._apply_common_style(ax, "Frequency")
                self._set_title_and_subtitle(ax, title)

                figs.append(fig)

        if multi_plot:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)
            idx = 0
            for i, rfacet in enumerate(self.row_facet):
                for j, cval in enumerate(col_vals):
                    if idx < len(figs):
                        tmp_ax = figs[idx].axes[0]
                        for artist in tmp_ax.get_children():
                            artist.remove()
                        sns.histplot(subset[rfacet].dropna(), bins=max_bins, ax=axes[i][j])
                        idx += 1
            plt.tight_layout()
            return fig

        return figs, num_cols
