import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_histogram(df, col_facet=None, row_facet=None, hue_col=None,
                   max_bins=10, multi_plot=True, **kwargs):
    """
    Plot histograms arranged by col_facet and row_facet.

    Ensures consistent x-axis scaling for all plots of the same row_facet.

    Args:
        df (pd.DataFrame): Prepared dataframe.
        col_facet (str): Column for splitting plots by column.
        row_facet (list): Columns to plot histograms of (treated as rows in layout).
        hue_col (str): Not used here but kept for API consistency.
        max_bins (int): Number of histogram bins.
        multi_plot (bool): True = single grid figure, False = list of individual figures.

    Returns:
        matplotlib.figure.Figure or list of Figures
    """
    row_facet = row_facet or []

    if not row_facet:
        fig, ax = plt.subplots(figsize=(6,4))
        if col_facet and pd.api.types.is_numeric_dtype(df[col_facet]):
            sns.histplot(df[col_facet].dropna(), bins=max_bins, ax=ax)
            ax.set_title(f"Histogram of {col_facet}")
            return fig
        return None

    col_vals = df[col_facet].unique() if col_facet else [None]
    num_rows = len(row_facet)
    num_cols = len(col_vals)

    # Compute global xlims for each row_facet column
    xlims = {}
    for rfacet in row_facet:
        all_data = df[rfacet].dropna()
        if not all_data.empty:
            xlims[rfacet] = (all_data.min(), all_data.max())

    if multi_plot:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False)
        axes = np.atleast_2d(axes)

        for i, rfacet in enumerate(row_facet):
            for j, cval in enumerate(col_vals):
                ax = axes[i][j]
                subset = df
                if cval is not None:
                    subset = subset[subset[col_facet] == cval]

                if subset[rfacet].dropna().empty:
                    ax.set_visible(False)
                    continue

                sns.histplot(subset[rfacet].dropna(), bins=max_bins, ax=ax)
                ax.set_xlim(xlims.get(rfacet))  # Ensure consistent x scale
                title = f"{rfacet}"
                if cval is not None:
                    title += f" | {col_facet}={cval}"
                ax.set_title(title)

        plt.tight_layout()
        return fig

    else:
        figs = []
        for rfacet in row_facet:
            xlim = xlims.get(rfacet)
            for cval in col_vals:
                subset = df
                if cval is not None:
                    subset = subset[subset[col_facet] == cval]

                if subset[rfacet].dropna().empty:
                    continue

                fig, ax = plt.subplots(figsize=(6,4))
                sns.histplot(subset[rfacet].dropna(), bins=max_bins, ax=ax)
                ax.set_xlim(xlim)  # Consistent x scale for this rfacet
                title = f"{rfacet}"
                if cval is not None:
                    title += f" | {col_facet}={cval}"
                ax.set_title(title)
                figs.append(fig)
        return figs, num_cols

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)

def weighted_agg(subset, group_keys, agg_func, agg_col, weight_col):
    if agg_func == "count":
        return subset.groupby(group_keys)[weight_col].sum().reset_index(name="value")
    elif agg_func == "sum":
        def wsum(x): return np.sum(x[agg_col] * x[weight_col])
        return subset.groupby(group_keys).apply(wsum).reset_index(name="value")
    elif agg_func == "mean":
        def wmean(x): return np.average(x[agg_col], weights=x[weight_col])
        return subset.groupby(group_keys).apply(wmean).reset_index(name="value")
    elif agg_func == "std":
        def wstd(x): return np.sqrt(np.cov(x[agg_col], aweights=x[weight_col]))
        return subset.groupby(group_keys).apply(wstd).reset_index(name="value")
    else:
        raise ValueError(f"Unsupported agg_func: {agg_func}")

def plot_groupedbarplot(df, col_facet=None, row_facet=None, hue_col=None,
                        agg_func="count", agg_col=None,
                        max_bins=10, multi_plot=True,
                        category_order=None,
                        weighting_mode="none", weight_col=None, **kwargs):
    logger.info(f"plot_groupedbarplot called with weighting_mode={weighting_mode}, weight_col={weight_col}")

    row_facet = row_facet or []
    if not row_facet:
        return None

    col_vals = df[col_facet].unique() if col_facet else [None]
    num_rows = len(row_facet)
    num_cols = len(col_vals)

    figs = []
    plot_data = []
    y_max_per_row = {}

    effective_row_facet = list(row_facet)

    if weighting_mode == "additional":
        effective_row_facet += [f"Weighted: {r}" for r in row_facet]

    for rfacet in effective_row_facet:
        is_weighted = rfacet.startswith("Weighted: ")
        base_rfacet = rfacet.replace("Weighted: ", "") if is_weighted else rfacet
        logger.debug(f"Processing rfacet={rfacet} (base={base_rfacet}) weighted={is_weighted}")

        is_numeric = pd.api.types.is_numeric_dtype(df[base_rfacet])
        if is_numeric:
            df[base_rfacet] = pd.qcut(df[base_rfacet], q=max_bins, duplicates="drop")
            df[base_rfacet] = df[base_rfacet].apply(lambda x: f"{x.left:.2f}-{x.right:.2f}")

        y_label = f"{agg_func} [{agg_col}]" if agg_func != "count" else f"{agg_func}"

        y_vals = []

        for cval in col_vals:
            subset = df
            if cval is not None:
                subset = subset[subset[col_facet] == cval]

            if subset.empty:
                continue

            group_keys = [base_rfacet]
            if hue_col:
                group_keys.append(hue_col)
            group_keys = list(dict.fromkeys(group_keys))

            df_plot = None

            if weighting_mode == "weighted only" or is_weighted:
                df_plot = weighted_agg(subset, group_keys, agg_func, agg_col, weight_col)
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

    if multi_plot:
        fig, axes = plt.subplots(len(row_facet), num_cols, figsize=(5*num_cols, 4*len(row_facet)), squeeze=False)
        axes = np.atleast_2d(axes)

        handles, labels = None, None

        for i, rfacet in enumerate(row_facet):
            for j, cval in enumerate(col_vals):
                ax = axes[i][j]

                # Find matching plot_data for this facet/col
                matching = [item for item in plot_data if item[0] == rfacet and item[1] == cval]
                if not matching:
                    ax.set_visible(False)
                    continue

                _, _, df_plot = matching[0]

                order = category_order.get(rfacet) if category_order and rfacet in category_order else None
                hue_order = category_order.get(hue_col) if hue_col and category_order else None

                sns.barplot(data=df_plot, x=rfacet, y="value", hue=hue_col, ax=ax,
                            order=order, hue_order=hue_order)

                if weighting_mode == "combined" and weight_col:
                    df_weighted = weighted_agg(subset, group_keys, agg_func, agg_col, weight_col)
                    sns.barplot(data=df_weighted, x=rfacet, y="value", hue=hue_col, ax=ax,
                                order=order, hue_order=hue_order, alpha=0.4, legend=False)

                main_title = f"{rfacet}" + (f" | {col_facet}={cval}" if cval else "")
                subtitle = ""
                if weighting_mode == "combined" and weight_col:
                    subtitle = f"Weighted overlay shown (alpha=0.4) using '{weight_col}'"

                # Set bold main title
                ax.set_title(main_title, fontweight='bold', fontsize='medium', loc='center')

                if subtitle:
                    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                            fontsize='small', ha='center', va='bottom')

            
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylabel(y_label)

                if rfacet in y_max_per_row:
                    ax.set_ylim(0, y_max_per_row[rfacet])

                if hue_col and handles is None:
                    handles, labels = ax.get_legend_handles_labels()
                if hue_col:
                    ax.get_legend().remove()

        if hue_col and handles:
            fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

        plt.tight_layout()
        return fig

    else:
        for rfacet in effective_row_facet:
            for cval in col_vals:
                matching = [item for item in plot_data if item[0] == rfacet and item[1] == cval]
                if not matching:
                    continue

                _, _, df_plot = matching[0]

                fig, ax = plt.subplots(figsize=(6, 4))

                order = category_order.get(rfacet) if category_order and rfacet in category_order else None
                hue_order = category_order.get(hue_col) if hue_col and category_order else None

                sns.barplot(data=df_plot, x=rfacet.replace("Weighted: ", ""), y="value", hue=hue_col, ax=ax,
                            order=order, hue_order=hue_order)

                if weighting_mode == "combined" and weight_col and not rfacet.startswith("Weighted: "):
                    df_weighted = weighted_agg(subset, group_keys, agg_func, agg_col, weight_col)
                    sns.barplot(data=df_weighted, x=rfacet, y="value", hue=hue_col, ax=ax,
                                order=order, hue_order=hue_order, alpha=0.4, legend=False)

                main_title = f"{rfacet}" + (f" | {col_facet}={cval}" if cval else "")
                subtitle = ""
                if weighting_mode == "combined" and weight_col:
                    subtitle = f"Weighted overlay shown (alpha=0.4) using '{weight_col}'"

                ax.text(0.5, 1.08, main_title,
                        transform=ax.transAxes,
                        fontsize='medium', fontweight='bold',
                        ha='center', va='bottom')

                if subtitle:
                    ax.text(0.5, 1.05, subtitle,
                            transform=ax.transAxes,
                            fontsize='small', ha='center', va='top')
                    
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylabel(y_label)

                max_y = y_max_per_row.get(rfacet, None)
                if max_y:
                    ax.set_ylim(0, max_y)

                if hue_col:
                    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

                figs.append(fig)

        return figs, num_cols
