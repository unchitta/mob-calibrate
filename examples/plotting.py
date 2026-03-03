import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




""" plotting.py

v2025.02.16
@unchitta

Helper functions for plotting sequence clusters

Use cluster_tempogram() or cluster_tempogram_weighted()
for plotting tempograms of sequences within different clusters

See Jupyter notebook for example.
"""



def cluster_tempogram(data, cluster_labels, colors, xticks, xlabs, legend_labels, suptitle=None):

    """
    Plot sequence tempogram for each cluster.

    Keyword arguments:
    ------------------
        data : pandas.DataFrame or numpy.array
            sequence data
        cluster_labels : list-like 
            list-like object containing cluster label 
            in the same order as the sequence rows in data
        colors : list
            colors to use for states in the sequences
            the order of the colors follows the sorted state names in labels_lookup
        xticks : list
            x steps to show in the plot
        xlabs : list
            labels for the x ticks (should be same length as xticks)
        legend_labels : dict
            lookup of the state names to show in the legend
        suptitle : str
            suptitle for all cluster-tempogram subplots
    """

    from copy import deepcopy

    temp = deepcopy(pd.DataFrame(data))
    temp['cluster'] = list(cluster_labels)

    # initialize subplots with 2 columns and nrows based on num clusters
    # if num clusters is an odd number, last subplot is removed
    nc = temp['cluster'].nunique()
    nrows = int(np.ceil(nc/2))
    fig, axes = _init_fig(nrows, nc % 2 != 0, sharex=True, sharey=True, suptitle=suptitle)

    # iterate through clusters and plot tempogram
    j=0
    for cluster_name, cluster_df in (temp.groupby('cluster')):

        title=f"Cluster {cluster_name}, N={len(cluster_df)}"

        (cluster_df
            .drop(columns=['cluster'])
            .melt(var_name='t', value_name='activity') # convert into long form
            .groupby('t', sort=False)['activity']   # use sort=False in case of string column names
            .value_counts(normalize=True, dropna=False)
            .unstack('activity')
            .plot.bar(stacked=True, color=colors, edgecolor='white', ax=axes[j])
        )

        _format_ax(axes[j], title, 't', 'Proportion', xticks, xlabs)
        j+=1

    axes[1].legend(legend_labels, bbox_to_anchor=(1.1, 1), fontsize='large')
    plt.tight_layout()



def cluster_tempogram_weighted(data, cluster_labels, weights, colors, xticks, xlabs, legend_labels, suptitle=None):

    """
    Plot sequence tempogram for each cluster.

    Keyword arguments:
    ------------------
        data : pandas.DataFrame or numpy.array
            sequence data
        cluster_labels : list-like 
            list-like object containing cluster labels
            in the same order as the sequence rows in data
        weights : list-like
            list-like object containing weights in the same order 
            as the sequence rows in data. Weights will be used in the
            calculations of proportions.
        colors : list
            colors to use for states in the sequences
            the order of the colors follows the sorted state names in legend_labels
        xticks : list
            x steps to show in the plot
        xlabs : list
            labels for the x ticks (should be same length as xticks)
        legend_labels : list
            names of the states to show in the legend
        suptitle : str
            suptitle for all cluster-tempogram subplots
    """

    from copy import deepcopy

    #if legend_labels is None:
    #    legend_labels = np.unique(data.values)
    temp = deepcopy(pd.DataFrame(data))
    temp['cluster'] = list(cluster_labels)
    temp['weight'] = weights

    colors_dict = dict(zip(legend_labels.values(), colors))

    # initialize subplots with 2 columns and nrows based on num clusters
    # if num clusters is an odd number, last subplot is removed
    nc = temp['cluster'].nunique()
    nrows = int(np.ceil(nc/2))
    fig, axes = _init_fig(nrows, nc % 2 != 0, sharex=True, sharey=True, suptitle=suptitle)
    axes[1].set_xlim(0)
    # iterate through clusters and plot tempogram
    j=0
    for cluster_name, cluster_df in (temp.groupby('cluster')):

        title=f"Cluster {cluster_name}, N={len(cluster_df)}"
        percent_states = {}

        # get total n to normalize by depending on if weights are used
        tot_n = cluster_df['weight'].sum()

        # convert seq into np array and convert into boolean matrices for each state
        seq = cluster_df.iloc[:,:-2].values
        for state, label in legend_labels.items():
            s = np.where(seq == state, 1, 0)
            s = s * cluster_df['weight'].values[:, None]
            s = s.sum(axis=0) / tot_n
            percent_states[label] = s
        
        # plot tempogram for this cluster
        bottom = np.zeros(seq.shape[1])
        for state_label, pct in percent_states.items():
            axes[j].bar(range(seq.shape[1]), pct, 
                        width=1, label=state_label, bottom=bottom, align='edge',
                        color=colors_dict[state_label], edgecolor='white')
            bottom += pct

        _format_ax(axes[j], title, 't', 'Proportion', xticks, xlabs)
        j+=1

    axes[1].legend(legend_labels.values(), bbox_to_anchor=(1.1, 1), fontsize='large')
    plt.tight_layout()



def _init_fig(nrows, odd=False, sharex=True, sharey=True, suptitle=None):
    fig, axes = plt.subplots(nrows, 2, 
                             figsize=(12, 3*nrows), dpi=300, 
                             sharey=sharey, sharex=sharex)
    if odd:
        axes[-1,-1].remove()
    axes = axes.flat
    fig.suptitle(suptitle, fontsize='x-large')
    return fig, axes


def _format_ax(ax, title, xlab=None, ylab=None, xticks=None, xlabs=None):
    ax.set_title(title, fontsize='x-large')
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabs, rotation=90)
    ax.set_xlabel(xlab, fontsize='x-large')
    ax.set_ylabel(ylab, fontsize='x-large')
    ax.tick_params(axis='both', labelsize='x-large')
    ax.legend()
    ax.get_legend().remove()
    return ax