import holoviews as hv
import hvplot.pandas


def plot_histogram(df, var, hue=None, bins=10, title=None):
    plot = df.hvplot.hist(var, by=hue, bins=bins)
    if title:
        plot = plot.opts(title=title)
    return plot


def plot_density(df, var, hue=None, title=None):
    plot = df.hvplot.kde(var, by=hue)
    if title:
        plot = plot.opts(title=title)
    return plot


def plot_sum(df, group_col, value_col, title=None):
    grouped = df.groupby(group_col)[value_col].sum()
    plot = grouped.hvplot.line()
    if title:
        plot = plot.opts(title=title)
    return plot


def generate_subplots(df, kind, var, hue=None, rowfacet=None, colfacet=None, bins=10):
    plots = []
    facets = []
    if rowfacet:
        facets.append(rowfacet)
    if colfacet:
        facets.append(colfacet)

    grouped = df.groupby(facets) if facets else [(None, df)]

    for keys, grp in grouped:
        label = f"{keys}" if keys else ""
        if kind == 'histogram':
            p = plot_histogram(grp, var=rowfacet, hue=hue, bins=bins, title=label)
        elif kind == 'density':
            p = plot_density(grp, var=rowfacet, hue=hue, title=label)
        elif kind == 'sum':
            p = plot_sum(grp, group_col=colfacet, value_col=rowfacet, title=label)
        else:
            p = hv.Text(0.5, 0.5, 'Unsupported plot type')
        plots.append(p)

    layout = hv.Layout(plots).cols(len(grouped) if colfacet else 1)
    return layout