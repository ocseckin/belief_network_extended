"""Belief network model."""
import math
import random
from itertools import combinations
from functools import reduce
from operator import mul
import networkx as nx
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
from bisect import bisect


def gnp_belief_network(n_nodes: int, prob: float, seed: int) -> nx.Graph:
    """create an ER random belief network (initialized with random beliefs)."""
    np.random.seed(seed)
    G = nx.gnp_random_graph(n_nodes, prob, seed=seed)
    initialize_with_random_beliefs(G, seed)
    return G


def initialize_with_random_beliefs(G: nx.Graph, seed: int):
    """initialize a belief network by assigning random weights (-1, 1)."""
    nx.set_edge_attributes(
        G, {edge: np.random.uniform(-1,1) for edge in G.edges()}, "belief"
    )


def complete_belief_network(N: int, edge_values="default") -> nx.Graph:
    """create a belief network which is fully connected and with all edges set to a user input (1) unifrom float value or (2) that specified by a list.
    If user does not specify any edge values, this outputs a fully connected random graph using initialize_with_random_beliefs (with seed = 0))"""
    G = nx.complete_graph(N)

    # if user does not input anything
    if edge_values == "default":
        initialize_with_random_beliefs(G, seed=0)
    else:
        if type(edge_values) == float or type(edge_values) == int:
            nx.set_edge_attributes(
                G, {edge: edge_values for edge in G.edges()}, "belief"
            )
        if type(edge_values) == list:
            nx.set_edge_attributes(
                G, {edge: edge_values[i] for i, edge in enumerate(G.edges())}, "belief"
            )
    return G

# added on June 14, 2024 by Ozgur

def color_fader(c1='blue',c2='red',mix=0): 
    #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def weight_gradient_color_matching(n=20, c1='red', c2='blue'):
    """Takes n as the number of intervals for the gradients 
        Returns a dictionary in which number of the interval is the key and value is the color"""

    gradient_colors = {}

    for i in range(n+1):
        gradient_colors[i] = color_fader(c1, c2, mix=i/n)
    
    return gradient_colors


def add_colors_to_edges(G, binary=True):
    """Takes the belief graph G, matches the edges of the network with colors and set them as attributes"""

    if binary:
        colors_for_edges = {}

        for e in G.edges():
            colors_for_edges[e] = {}
            if G.edges[e]['belief'] > 0:
                colors_for_edges[e]['color'] = 'blue'
            else:
                colors_for_edges[e]['color'] = 'red'
    
    else:
        gradient_colors = weight_gradient_color_matching(n=20, c1='red', c2='blue')
        n = len(gradient_colors)
        boundaries = np.linspace(-1,1,n+1)[1:]
        colors_for_edges = {e:{"color":gradient_colors[bisect(boundaries, G.edges[e]['belief'])]} for e in G.edges()}

    nx.set_edge_attributes(G, colors_for_edges)


def find_triads(G: nx.Graph):
    """Find the triads in a given network"""
    triads = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    return triads


def triad_energy(G: nx.Graph, triad, weight_key="belief") -> float:
    """Calculate the product of beliefs."""
    triadic_weights = [G[n1][n2][weight_key] for n1, n2 in combinations(triad, 2)]
    return reduce(mul, triadic_weights, 1)


def internal_energy(G: nx.Graph) -> float:
    """calcualte the internal energy given the belief network using social balance."""
    triads = find_triads(G)
    return -1.0 * sum(triad_energy(G, triad) for triad in triads) / len(triads)


def derivative_triad_energy(G: nx.Graph, triad, focal_edge, weight_key="belief") -> float:
    """Calculate the product of beliefs of a triad not including the focal edge
    Example: (e_int = a*b*c + a*d*e + d*e*f) and focal edge = a
    the derivative of internal energy = b*c + d*e
    """

    # the weights of the triadic edges not including the focal edge
    triad_edges = itertools.combinations(triad, 2)
    weights_wout_focal = [G[n1][n2][weight_key] for n1, n2 in triad_edges if set([n1, n2])!=set([focal_edge[0], focal_edge[1]])]

    return reduce(mul, weights_wout_focal, 1)