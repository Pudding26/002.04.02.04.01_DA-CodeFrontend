# plot_control.py
import panel as pn
import holoviews as hv
import pandas as pd
import hvplot.pandas

PLOTS = {
    'Barplot': {
        'Histogram': {
            'aggs': ['count'],
            'controls': {
                'col_facet': True,
                'row_facet': True,
                'hue': {'enabled': True, 'max_hue_cols': 5},
                'needs_bins': True
            }
        }
    },
    'Lineplot': {
        'Density': {
            'aggs': ['mean', 'sum'],
            'controls': {
                'col_facet': True,
                'row_facet': True,
                'hue': {'enabled': False},
                'needs_bins': False
            }
        }
    },
    'Scatterplot': {
        'Basic': {
            'aggs': ['none'],
            'controls': {
                'col_facet': True,
                'row_facet': True,
                'hue': {'enabled': True, 'max_hue_cols': 8},
                'needs_bins': False
            }
        }
    }
}

class PlotControls(pn.viewable.Viewer):
    def __init__(self, plots_area, **params):
        super().__init__(**params)
        self.plots_area = plots_area

        # Widgets
        self.plot_type = pn.widgets.Select(name='Plot Type', options=list(PLOTS.keys()))
        self.subplot_type = pn.widgets.Select(name='Subplot Type')
        self.hue_col = pn.widgets.Select(name='Hue / Color', options=[''])
        self.row_facet = pn.widgets.Select(name='Row Facet')
        self.col_facet = pn.widgets.Select(name='Column Facet')
        self.bins = pn.widgets.IntSlider(name='Bins', start=5, end=50, step=5, value=10)
        self.render_btn = pn.widgets.Button(name='Render Plot', button_type='primary')

        self.controls_panel = pn.Column(
            pn.Row(self.plot_type, self.subplot_type),
            pn.Row(self.hue_col, self.row_facet, self.col_facet),
            self.bins,
            self.render_btn
        )

        self._current_df = None
        self._setup_callbacks()

    def __panel__(self):
        return self.controls_panel

    def _setup_callbacks(self):
        self.plot_type.param.watch(self._update_subplots, 'value')
        self.subplot_type.param.watch(self._update_controls, 'value')
        self.render_btn.on_click(self._render_plot)
        self._update_subplots()

    def _update_subplots(self, *events):
        self.subplot_type.options = list(PLOTS[self.plot_type.value].keys())
        self.subplot_type.value = self.subplot_type.options[0]
        self._update_controls()

    def _update_controls(self, *events):
        cfg = PLOTS[self.plot_type.value][self.subplot_type.value]['controls']
        self.bins.visible = cfg.get('needs_bins', False)
        self.row_facet.visible = cfg.get('row_facet', False)
        self.col_facet.visible = cfg.get('col_facet', False)
        self.hue_col.visible = cfg.get('hue', {}).get('enabled', False)

    def update_column_options(self, df):
        self._current_df = df
        if not df.empty:
            cols = df.columns.tolist()
            self.hue_col.options = [''] + cols
            self.row_facet.options = cols
            self.col_facet.options = cols

    def _render_plot(self, *events):
        self.plots_area.clear()
        df = self._current_df
        if df is None or df.empty:
            self.plots_area.append(pn.pane.Markdown("⚠️ No data to plot."))
            return

        kind = self.plot_type.value
        sub_kind = self.subplot_type.value
        cfg = PLOTS[kind][sub_kind]['controls']

        hue = self.hue_col.value or None
        rowfacet = self.row_facet.value or None
        colfacet = self.col_facet.value or None
        bins = self.bins.value

        facets = []
        if colfacet: facets.append(colfacet)
        if rowfacet: facets.append(rowfacet)

        grouped = df.groupby(facets) if facets else [(None, df)]

        plots = []
        for keys, grp in grouped:
            label = ', '.join(f"{f}: {k}" for f, k in zip(facets, keys if isinstance(keys, tuple) else [keys])) if keys else ""
            plot = self._plot_single(grp, kind, sub_kind, rowfacet, hue, bins, label)
            plots.append(plot)

        layout = hv.Layout(plots).cols(len(grouped) if colfacet else 1)
        self.plots_area.append(pn.pane.HoloViews(layout))

    def _plot_single(self, df, kind, sub_kind, var, hue, bins, title):
        opts = dict(title=title, height=400, width=600)
        if kind == 'Barplot':
            return df.hvplot.hist(var, by=hue, bins=bins).opts(**opts)
        if kind == 'Lineplot' and sub_kind == 'Density':
            return df.hvplot.kde(var, by=hue).opts(**opts)
        if kind == 'Lineplot' and sub_kind == 'Sum':
            return df.groupby(var).size().hvplot.line().opts(**opts)
        if kind == 'Scatterplot':
            return df.hvplot.scatter(x=var, y=var, by=hue).opts(**opts)
        return hv.Text(0.5, 0.5, 'Unsupported plot')

def build_controls(plots_area):
    return PlotControls(plots_area)
