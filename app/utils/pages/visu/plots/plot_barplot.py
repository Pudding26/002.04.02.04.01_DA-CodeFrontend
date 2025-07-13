import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_histogram(df, col_facet, row_facet, hue_col, max_bins, multi_plot=True):
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
