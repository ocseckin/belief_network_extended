"""Belief network model."""
import math
import random
from itertools import combinations, product
from functools import reduce
from operator import mul
import networkx as nx
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
from bisect import bisect
from scipy.stats import norm


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


def complete_belief_network(n_nodes: int, edge_values="default") -> nx.Graph:
    """create a belief network which is fully connected and with all edges set to a user input (1) unifrom float value or (2) that specified by a list.
    If user does not specify any edge values, this outputs a fully connected random graph using initialize_with_random_beliefs (with seed = 0))"""
    G = nx.complete_graph(n_nodes)

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
    triad_edges = combinations(triad, 2)
    weights_wout_focal = [G[n1][n2][weight_key] for n1, n2 in triad_edges if set([n1, n2])!=set([focal_edge[0], focal_edge[1]])]

    return reduce(mul, weights_wout_focal, 1)

def derivative_internal_energy(G: nx.Graph, focal_edge) -> float:

    """calculate the derivative of the internal energy of an individual
    with respect to the focal edge, evaluated at the given weight configuration.
    This is reflects how much the internal energy changes upon changing the focal
    edge weight.

    Parameters
    ----------
    G -> networkx graph of the individual belief network

    focal_edge -> focal "receiver edge" tuple eg: (1, 2)

    Returns
    -------
    float
        derivative of internal evaluated at the given weight configuration (edge weights)
    """
    # The triads not including the ones without the focal edge
    triads = find_triads(G)
    triads_with_focal_edge = [t for t in triads if (focal_edge[0] in t) & (focal_edge[1] in t)]
    
    return -1.0 * sum(derivative_triad_energy(G, triad, focal_edge) for triad in triads_with_focal_edge)


def community_social_network(N: int, mu: float, M: int, seed=None):

    """Obtain the social network of two communities with half of the nodes in c1 and half in c2.
    - probability of intra community edges = (1-mu)
    - probability of inter community edges = mu

    Parameters
    ----------
    - N -> Number of nodes of the social network
    - mu -> parameter to calculate probabilities of inter and intra community edges
    - M -> Number of edges of the social network

    Returns
    -------
    - The Social Network -> nx.Graph()
    - list of nodes in community 1
    - list of nodes in community 2

    """

    # fraction of intra community links
    intra = math.ceil((1 - mu) * M)

    # fraction of inter community links
    inter = M - intra

    G = nx.Graph()
    G.add_nodes_from(list(range(0, N)))

    if seed is None:
        random.seed()
    else:
        random.seed(seed)

    # community 1 - half the nodes
    comm1 = random.sample(list(range(0, N)), k=math.ceil(N / 2))
    # community 2 - the remaining nodes
    comm2 = list(set(range(0, N)) - set(comm1))

    # intra community edges
    G.add_edges_from(
        random.sample(
            list(combinations(comm1, 2))
            + list(combinations(comm2, 2)),
            intra,
        )
    )
    # inter community edges
    G.add_edges_from(random.sample(list(product(comm1, comm2)), inter))

    return G, comm1, comm2


def choose_sender_receiver_belief(G):
    """Randomly chooses 
    1. a sender node 
    2. a receiver node from the sender's neighbors 
    3. a belief (focal edge) that will be sent from sender to receiver.

    Parameters
    ----------
    - G -> social network
    
    Returns
    -------
    - sender
    - receiver
    - focal_edge"""

    sender = random.sample([*G.nodes()],1)[0]
    receiver = random.sample([*nx.all_neighbors(G, sender)], 1)[0]

    focal_edge = random.sample([*G.nodes[sender]['belief_network'].edges()], 1)[0]

    return sender, receiver, focal_edge


def stochasticity(mean, normal_scale = 0.2):
    """Takes a mean and standard deviation value and returns a randomly chosen value from
    a normal distribution"""
    
    return list(norm.rvs(loc=mean, scale=normal_scale, size=1))[0]


def calculate_updated_weight(G, sender, receiver, focal_edge, alpha=1.5, beta=1):
    """Calculates the updated weight of the belief (focal edge)
    
     Parameters
    ----------
    - Social network
    - Sender's belief
    - Receiver's belief
    - Focal edge (belief)
    - Alpha
    - Beta
    
    Returns
    -------
    - Updated receiver belief (b_i_plus_1)
    """
    
    b_i = G.nodes[receiver]['belief_network'].edges[focal_edge].get('belief')
    b_j = G.nodes[sender]['belief_network'].edges[focal_edge].get('belief')
    
    first_term = alpha * b_j

    derivative = -1 * derivative_internal_energy(G.nodes[receiver]['belief_network'], focal_edge=focal_edge)
    second_term = beta * derivative

    update_term = first_term + second_term

    b_i_plus_1 = b_i + stochasticity(update_term, normal_scale = 0.2)

    if b_i_plus_1 > 1:
        b_i_plus_1 = 1
    elif b_i_plus_1 < -1:
        b_i_plus_1 = -1

    return b_i_plus_1


def embed_b_i_plus_1_to_belief_network(G, receiver, focal_edge, b_i_plus_1):
    """Embeds the updated receiver belief into its belief network inside the social 
    network."""
    
    G.nodes[receiver]['belief_network'].edges[focal_edge]['belief'] = b_i_plus_1