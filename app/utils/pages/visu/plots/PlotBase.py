# PlotBase.py
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger("plotbase")


class PlotBase:
    def __init__(self, df, col_facet=None, row_facet=None, hue_col=None, category_order=None):
        self.df = df
        self.col_facet = col_facet
        self.row_facet = row_facet or []
        self.hue_col = hue_col
        self.category_order = category_order or {}

        # Generic init debug
        self._debug_log_init()

    # -------------------------------
    # Debug helper
    # -------------------------------
    def _debug_log_init(self):
        logger.info(
            "[PlotBase Init] "
            f"df.shape={self.df.shape}, "
            f"cols={list(self.df.columns)}, "
            f"col_facet={self.col_facet}, "
            f"row_facet={self.row_facet}, "
            f"hue_col={self.hue_col}, "
            f"category_order_keys={list(self.category_order.keys()) if self.category_order else []}"
        )

    def _debug_log_subset(self, msg, df_subset):
        logger.info(
            f"[PlotBase Subset] {msg} "
            f"shape={df_subset.shape}, "
            f"non_empty_cols={[c for c in df_subset.columns if not df_subset[c].isnull().all()]}"
        )

    # -------------------------------
    # Facet & ordering helpers
    # -------------------------------
    def _get_col_vals(self):
        """Return unique column facet values (or [None] if not set)."""
        vals = self.df[self.col_facet].unique() if self.col_facet else [None]
        logger.info(f"[PlotBase] _get_col_vals({self.col_facet}) → {vals}")
        return vals

    def _get_order(self, key):
        """Return order for a given key if specified in category_order."""
        order = self.category_order.get(key) if key and self.category_order else None
        logger.info(f"[PlotBase] _get_order({key}) → {order}")
        return order

    # -------------------------------
    # Styling helpers
    # -------------------------------
    def _apply_common_style(self, ax, ylabel=None, max_y=None):
        """Apply consistent axis styling."""
        if max_y is not None:
            ax.set_ylim(0, max_y)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(ax.get_xlabel(), fontsize="small")
        ax.set_ylabel(ax.get_ylabel(), fontsize="small")
        ax.tick_params(axis="both", which="major", labelsize="small")

    def _set_title_and_subtitle(self, ax, title, subtitle=None):
        """Apply standardized title + optional subtitle."""
        ax.text(
            0.5, 1.08, title,
            transform=ax.transAxes,
            fontsize="medium", fontweight="bold",
            ha="center", va="bottom"
        )
        if subtitle:
            ax.text(
                0.5, 1.05, subtitle,
                transform=ax.transAxes,
                fontsize="small", ha="center", va="top"
            )

    # -------------------------------
    # Abstract API
    # -------------------------------
    def plot(self, *args, **kwargs):
        """Each child must implement its own plot() method."""
        raise NotImplementedError("Subclasses must implement plot()")
    