import math
import random
from itertools import combinations, product
from functools import reduce
from operator import mul
import networkx as nx
import numpy as np
from scipy.stats import norm

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
    
    communities = {n:"democrat" for n in comm1}
    communities.update({n:"republican" for n in comm2})

    return G, communities


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
    
    if sender == None:
        # if no sender, take b_j as 0
        b_j = 0
    else:
        b_j = G.nodes[sender]['belief_network'].edges[focal_edge].get('weight')

    b_i = G.nodes[receiver]['belief_network'].edges[focal_edge].get('weight')
        
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


def find_mutual_edges(G, H):
    
    # find the mutual edges between two networks
    common_edges = set(map(frozenset, G.edges())) & set(map(frozenset, H.edges()))

    # convert the result back to a list of tuples if needed
    common_edges_as_tuples = [tuple(edge) for edge in common_edges]
    
    return common_edges_as_tuples


def choose_sender_receiver_focal_edge(G, mutual = False):
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

    # find the mutual edges
    if mutual:
        mutual_edges = find_mutual_edges(G.nodes[sender]['belief_network'], G.nodes[receiver]['belief_network'])
        # choose one of the mutual edges as the focal edge
        index = np.random.randint(len(mutual_edges))
        focal_edge = mutual_edges[index]
        
    else:
        focal_edge = random.choice([*G.nodes[sender]['belief_network'].edges()])

    return sender, receiver, focal_edge


def embed_b_i_plus_1_to_belief_network(G, receiver, focal_edge, b_i_plus_1):
    """Embeds the updated receiver belief into its belief network inside the social 
    network."""
    
    G.nodes[receiver]['belief_network'].edges[focal_edge]['weight'] = b_i_plus_1

    
def generate_embed_belief_networks_to_social_network(social_G, communities):
    
    belief_networks = {}

    for n in social_G.nodes():

        # initialize the graph
        belief_G = nx.Graph()

        # add social ties
        belief_G.add_weighted_edges_from([(n, neighbor, np.random.normal(loc=0, scale=.2)) for neighbor in nx.neighbors(social_G, n)])

        # add democrat/republican nodes
        comm = communities[n]
        opp_comm = {"republican":"democrat", "democrat":"republican"}[comm]

        # add community edge
        belief_G.add_edge(n, comm, weight=1)
        belief_G.add_edge(n, opp_comm, weight=-1)

        # add the concept of a "truck"
        belief_G.add_edge(n, "truck", weight=np.random.normal(loc=0, scale=.2))

        # what are all the possible edges
        all_possible_edges = set([*combinations([*belief_G.nodes()], r=2)])

        # which ones are non-existent?
        nonexistent_edges = set(map(frozenset, all_possible_edges)) - set(map(frozenset, belief_G.edges()))

        # add very small weight to all the non-existent edges
        belief_G.add_weighted_edges_from([(tuple(e)[0], tuple(e)[1], np.random.normal(loc=0, scale=.00001)) for e in nonexistent_edges])

        # add belief network to the dictionary to update the social network after
        belief_networks[n] = {}
        belief_networks[n]['belief_network'] = belief_G     

    # set node attributes accordingly
    nx.set_node_attributes(social_G, belief_networks)
    
    return social_G


def find_triads(G: nx.Graph):
    """Find the triads in a given network"""
    triads = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    return triads


def find_triads_with_focal_edge(triads, focal_edge):
    
    triads_with_focal_edge = [t for t in triads if (focal_edge[0] in t) & (focal_edge[1] in t)]
    
    return triads_with_focal_edge


def derivative_triad_energy(G: nx.Graph, triad, focal_edge, weight_key="weight") -> float:
    """Calculate the product of beliefs of a triad not including the focal edge
    Example: (e_int = a*b*c + a*d*e + d*e*f) and focal edge = a
    the derivative of internal energy = b*c + d*e
    """

    # the weights of the triadic edges not including the focal edge
    triad_edges = combinations(triad, 2)
    weights_wout_focal = [G[n1][n2][weight_key] for n1, n2 in triad_edges if set([n1, n2])!=set([focal_edge[0], focal_edge[1]])]

    return reduce(mul, weights_wout_focal, 1)


def derivative_internal_energy(G: nx.Graph, focal_edge) -> float:

    """Calculate the derivative of the internal energy of an individual
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
    
    triads_with_focal_edge = find_triads_with_focal_edge(triads, focal_edge)
    
    return -1.0 * sum(derivative_triad_energy(G, triad, focal_edge) for triad in triads_with_focal_edge)


def stochasticity(mean, normal_scale = 0.2):
    """Takes a mean and standard deviation value and returns a randomly chosen value from
    a normal distribution"""
    
    return list(norm.rvs(loc=mean, scale=normal_scale, size=1))[0]