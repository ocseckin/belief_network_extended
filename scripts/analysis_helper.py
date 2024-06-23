import sys
sys.path.insert(1, '../scripts')
import extended_model
import importlib
import glob
import json
importlib.reload(extended_model)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import bisect

import itertools
from functools import reduce
from operator import mul
from scipy.stats import norm
import random
from math import comb

import multiprocessing as mp

from tqdm import tqdm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14

def overall_analysis(n_nodes, track, i, save=False):

    types_of_stable = extended_model.permute_stable_networks(n_nodes)

    fig, axs = plt.subplots(2, 2, figsize=(14,8))
    mean_values = np.array([np.mean(v['internal_energies']) for v in track.values()])
    std_values = np.array([np.std(v['internal_energies']) for v in track.values()])/2

    x = [*track.keys()]

    axs[0][0].plot(x, mean_values, label = 'Avg.')
    axs[0][0].fill_between(x, mean_values-std_values, mean_values+std_values, facecolor='gray', alpha=0.3, label = 'Std.')

    #for sign, label in zip([-1,1], ['Better-off', 'Worse-off']):
    #    axs[1][0].plot([k for k,v in track.items() if 'better_off' in v.keys()],
    #                [sum(v['better_off']==sign) for k,v in track.items() if 'better_off' in v.keys()],
    #                label=label,
    #                )

    # Plot the number of stable agents
    for stable in types_of_stable:
        axs[1][0].plot(x,
                    [sum([all(b==stable) for b in v['beliefs']]) for v in track.values()],
                    label=", ".join([str(i) for i in list(stable)]),
                    )
    axs[1][0].plot(x,
                [sum(v['internal_energies']==-1) for k,v in track.items()],
                label='Total Stable',
                color='black',
                linestyle='--',
                alpha=.5)

    # Plot the count of unique belief systems
    axs[0][1].plot(x,
                    [len(np.unique(v['beliefs'], axis=0))  for v in track.values()],
                    label='Unique Belief Networks')

    # calculate the number of existing stable cases
    # e.g. if [1,1,1] and [-1,-1,1] exist but the other stable cases don't, the count 
    # of stable cases is 2
    existing_stable_cases = np.sum(np.array([[1 if len([b for b in v['beliefs'] if all(b == s)]) != 0 else 0 for v in track.values()] for s in types_of_stable]), axis=0)
    axs2 = axs[0][1].twinx()
    axs2.plot(x,
            existing_stable_cases,
            color='gray',
            label='Stable Belief Networks'
            )
    axs2.set_ylabel('Unique Stable Belief Network Count')
    axs2.legend(frameon=False, bbox_to_anchor=(0., .82, 1., .102))

    # p-index
    beliefs = [v['beliefs'] for v in track.values()]

    from sklearn.cluster import DBSCAN
    clustering = [DBSCAN(eps=.1, min_samples=3).fit(b).labels_ for b in beliefs]

    polarization_analysis = []

    for _, arr in enumerate(beliefs):
        cluster_centroids, polarization = compute_polarization(clustering = clustering[_], belief_arr = arr)
        polarization_analysis.append(polarization)

    axs[1][1].plot([i*20 for i in range(len(polarization_analysis))],
                    polarization_analysis)
    axs[1][1].set_ylim(-.05,1.05)

    for row, col, ylabel in zip([0, 1, 1, 0, 1], [0, 0, 0, 1, 1], ['Internal Energy', 'Agent Count', 'Agent Count', 'Unique Belief Network Count', 'p-index']):
        axs[row][col].grid(alpha=.3)
        axs[row][col].set_ylabel(ylabel)
        axs[row][col].legend(frameon=False)

    axs[1][0].set_xlabel('Time Step (t)')
    axs[1][1].set_xlabel('Time Step (t)')

    fig.tight_layout()
    
    if save:
        fig.savefig(f'../figures/main/simulation_{i}.pdf')
    fig.show()


def node_coloring_prep(types_of_stable, colors = ["#ef476f","#ffd166","#118ab2","#073b4c"]):

    types_of_stable_naming = {n:s for n,s in zip([",".join([str(i) for i in s]) for s in types_of_stable], types_of_stable)}
    types_of_stable_coloring = {k:colors[i] for i,k in enumerate(types_of_stable_naming.keys())}

    return types_of_stable_naming, types_of_stable_coloring


def node_coloring_on_social_network(G, types_of_stable_naming, types_of_stable_coloring):

    node_coloring = {}

    for node in G.nodes():
        
        # initialize the node in the dictionary
        node_coloring[node] = {}
        node_coloring[node]['color'] = 'gray'

        node_coloring[node]
        
        # extract the belief network
        belief_network = G.nodes[node]['belief_network']
        
        # extract the belief weights
        belief_weights = [belief_network.edges[e]['belief'] for e in belief_network.edges()]

        # transform into np array
        belief_weights = np.array(belief_weights)

        for n,s in types_of_stable_naming.items():
            if all(belief_weights == s):
                node_coloring[node]['color'] = types_of_stable_coloring[n]            

    return node_coloring


def compute_polarization(clustering, belief_arr):
    
    # keys as cluster labels and values as the nodes in them
    cluster_node_match = {l:np.where(clustering==l)[0] for l in np.unique(clustering) if l!=-1}

    # compute the centroids of the clusters
    cluster_centroids = {k:np.median(np.array([belief_arr[n] for n in v]), axis=0) for k, v in cluster_node_match.items()}
    
    # measure polarization by computing the variance among cluster centroids
    if len(cluster_centroids.keys())<=1:
        polarization = 0
    else:
        polarization = np.var(np.array([*cluster_centroids.values()]))

    return cluster_centroids, polarization

