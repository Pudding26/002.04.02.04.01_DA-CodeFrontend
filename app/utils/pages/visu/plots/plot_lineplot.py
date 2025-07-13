import matplotlib.pyplot as plt
import seaborn as sns

def plot_density(df, col_facet, row_facet, hue_col, max_bins):
    fig, ax = plt.subplots(figsize=(8, 4))
    if hue_col:
        for key, grp in df.groupby(hue_col):
            sns.kdeplot(grp[col_facet], label=str(key), ax=ax)
        ax.legend(title=hue_col)
    else:
        sns.kdeplot(df[col_facet], ax=ax)
    ax.set_title("Density Plot")
    return fig

def plot_sum(df, col_facet, row_facet, hue_col, max_bins):
    if not col_facet:
        return None

    group_keys = [col_facet]
    if hue_col:
        group_keys.append(hue_col)
    if row_facet:
        group_keys.extend(row_facet)

    df_grouped = df.groupby(group_keys).size().reset_index(name="value")

    fig, ax = plt.subplots(figsize=(8, 4))
    if hue_col:
        for key, grp in df_grouped.groupby(hue_col):
            ax.plot(grp[col_facet], grp["value"], label=str(key))
        ax.legend(title=hue_col)
    else:
        ax.plot(df_grouped[col_facet], df_grouped["value"])
    ax.set_title("Sum Plot")
    return fig
