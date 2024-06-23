"""Belief network model."""
import analysis_helper
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
from math import comb
from tqdm import tqdm
from sklearn.cluster import DBSCAN


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
    """Calculate the internal energy given the belief network using social balance."""
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


def internal_energy_analysis(results):
    """Takes the results coming out of N simulations, outputs their average"""
    
    simulation_count = len(results)
    max_T_simulation = max([len(results[i].keys()) for i in range(simulation_count)])

    internal_energy_analysis_data = {}

    for k in results.keys():

        # take the average of internal energies throughout each step of the simulation
        avg_internal_energies = np.mean(np.array([v['internal_energies'] for v in results[k].values()]), axis=1)

        # concatenate with an array full of -1s to make the shapes compatible
        avg_internal_energies = np.concatenate([avg_internal_energies, np.array((max_T_simulation - len(avg_internal_energies)) * [-1])])

        # log
        internal_energy_analysis_data[k] = avg_internal_energies

    x = [*range(np.array([*internal_energy_analysis_data.values()]).shape[1])]
    avg = np.mean(np.array([*internal_energy_analysis_data.values()]),axis=0)
    upper = np.percentile(np.array([*internal_energy_analysis_data.values()]),q=97.5,axis=0)
    lower = np.percentile(np.array([*internal_energy_analysis_data.values()]),q=2.5,axis=0)

    internal_energy_analysis_data_sum = {}
    internal_energy_analysis_data_sum['x'] = x
    internal_energy_analysis_data_sum['avg'] = avg
    internal_energy_analysis_data_sum['upper'] = upper
    internal_energy_analysis_data_sum['lower'] = lower
    
    return internal_energy_analysis_data_sum


def better_off_worse_off_analysis(results):
    """Takes the average of the count of better-off and worse-off agents in each step of the simulation"""

    better_off_worse_off_data = {}
    max_iteration = []

    for sim_no, v in results.items():
        better_off_worse_off_data[sim_no] = {}

        for i, l in zip([-1,1], ['better_off','worse_off']):
            count = [len(np.where(d['better_off']==i)[0]) for t, d in v.items() if 'better_off' in d.keys()]
            better_off_worse_off_data[sim_no][l] = count

            iteration = [t for t,d in v.items() if 'better_off' in d.keys()]
            if len(iteration) > len(max_iteration):
                max_iteration = iteration

    # concatenation
    better_off_worse_off_data = {k:{'better_off':np.concatenate([np.array(v['better_off']), np.array([0]*(len(max_iteration)-len(v['better_off'])))]), 'worse_off': np.concatenate([np.array(v['worse_off']), np.array([0]*(len(max_iteration)-len(v['better_off'])))]) } for k,v in better_off_worse_off_data.items()}

    x = max_iteration
    better_off_worse_off_data_sum = {}

    for l in ['better_off', 'worse_off']:

        avg = np.mean(np.array([v[l] for v in better_off_worse_off_data.values()]), axis=0)
        upper = np.percentile(np.array([v[l] for v in better_off_worse_off_data.values()]),q=97.5, axis=0)
        lower = np.percentile(np.array([v[l] for v in better_off_worse_off_data.values()]), q=2.5, axis=0)

        better_off_worse_off_data_sum[l] = {}
        better_off_worse_off_data_sum[l]['x'] = x
        better_off_worse_off_data_sum[l]['avg'] = avg
        better_off_worse_off_data_sum[l]['upper'] = upper
        better_off_worse_off_data_sum[l]['lower'] = lower

    return better_off_worse_off_data_sum


def stability_analysis(results, n_nodes):

    types_of_stable = permute_stable_networks(n_nodes)
    n_edges = comb(n_nodes,2)

    stability_analysis_data = {}

    for k in results.keys():

        temp = {}

        for stable in types_of_stable:
            stable_name = ", ".join([str(i) for i in list(stable)])
            temp[stable_name] = [sum([all(b==stable) for b in v['beliefs']]) for v in results[k].values()]

        temp = {i:v for i,(k,v) in enumerate(sorted([(k,v) for k,v in temp.items()], key=lambda x: np.mean(x[1]), reverse=True))}
        stability_analysis_data[k] = temp

    max_T_simulation = max([len(v[0]) for v in stability_analysis_data.values()])

    stability_analysis_data = {k:{k_:v_+(max_T_simulation-len(v_))*[v_[-1]] for k_,v_ in v.items()} for k,v in stability_analysis_data.items()}

    stability_analysis_data_sum = {}

    for polarized in [*stability_analysis_data[0].keys()]:
        stability_analysis_data_sum[polarized] = {}
        stability_analysis_data_sum[polarized]['x'] = [*range(max_T_simulation)]
        stability_analysis_data_sum[polarized]['avg'] = np.array([v[polarized] for v in stability_analysis_data.values()]).mean(axis=0)
        stability_analysis_data_sum[polarized]['upper'] = np.percentile(np.array([v[polarized] for v in stability_analysis_data.values()]), axis=0, q=97.5)
        stability_analysis_data_sum[polarized]['lower'] = np.percentile(np.array([v[polarized] for v in stability_analysis_data.values()]), axis=0, q=2.5)
    
    return stability_analysis_data_sum


def unique_belief_count_analysis(results):

    # take only the belief arrays from each simulation
    belief_arrays = [[len(np.unique(v_['beliefs'], axis=0)) for v_ in v.values()] for k, v in results.items()]

    # find the maximum T in the dataset
    max_T_simulation = max([len(a) for a in belief_arrays])

    # padding for all belief arrays so that they are all the same shape
    belief_arrays = [a+[a[-1]]*(max_T_simulation-len(a)) for a in belief_arrays]

    # transform it into an array
    belief_arrays = np.array(belief_arrays)

    # summary statistics for each column
    avg = np.mean(belief_arrays, axis=0)
    upper = np.percentile(belief_arrays, axis=0, q=97.5)
    lower = np.percentile(belief_arrays, axis=0, q=2.5)
    x = [i*20 for i in range(len(belief_arrays[0]))]

    unique_belief_count_analysis_sum = {'x':x, 'avg':avg, 'upper':upper, 'lower':lower}
    
    return unique_belief_count_analysis_sum


def unique_stable_network_count_analysis(results, n_nodes):

    types_of_stable = permute_stable_networks(n_nodes)

    # find the unique number of stable beliefs in each simulation
    unique_stable_network_count = [np.sum(np.array([[1 if len([b for b in v['beliefs'] if all(b == s)]) != 0 else 0 for v in track.values()] for s in types_of_stable]), axis=0) for track in results.values()]

    # find the longest simulation time
    max_T = max([len(a) for a in unique_stable_network_count])

    # apply padding to make each array same shape
    unique_stable_network_count = np.array([np.concatenate([a, np.array([a[-1]]*(max_T-len(a)))]) for a in unique_stable_network_count])

    # compute summary statistics
    x = [i*20 for i in range(len(unique_stable_network_count[0]))]
    avg = np.mean(unique_stable_network_count,axis=0)
    upper = np.percentile(unique_stable_network_count, axis=0, q=97.5)
    lower = np.percentile(unique_stable_network_count, axis=0, q=2.5)

    # put all together in a dict
    unique_stable_network_count_sum = {'x':x, 'avg':avg, 'upper':upper, 'lower':lower}

    return unique_stable_network_count_sum


def polarization_analysis(results):
    
    polarization_analysis_data = {}

    for sim_no, track in results.items():
        beliefs = beliefs = [v['beliefs'] for v in track.values()]

        clustering = [DBSCAN(eps=.1, min_samples=3).fit(b).labels_ for b in beliefs]

        polarization_analysis = []

        for _, arr in enumerate(beliefs):
            cluster_centroids, polarization = analysis_helper.compute_polarization(clustering = clustering[_], belief_arr = arr)
            polarization_analysis.append(polarization)

        polarization_analysis_data[sim_no] = polarization_analysis

    # apply padding to have the same length of lists for each simulation
    max_T = max([len(e) for e in polarization_analysis_data.values()])
    polarization_analysis_data = {sim_no:e+[e[-1]]*(max_T-len(e)) for sim_no, e in polarization_analysis_data.items()}

    # transform the experiment data into an array
    arr = np.array([*polarization_analysis_data.values()])

    # get aggregated results
    x = [i*20 for i in range(max_T)]
    avg = np.mean(arr, axis=0)
    upper = np.percentile(arr, 97.5)
    lower = np.percentile(arr, 2.5)

    polarization_analysis_data_sum = {'x':x, 'avg':avg, 'upper':upper, 'lower':lower}
    
    return polarization_analysis_data_sum


def round_down_even(n):
    return 2 * int(n // 2)


def permute_stable_networks(n_nodes):

    # compute the number of edges
    n_edges = comb(n_nodes,2)

    # find the nearest even number that is equal or less than n_edges
    nearest_even = round_down_even(n_edges)

    # list of all even numbers until the nearest_even
    all_even = [*range(0, nearest_even, 2)] + [nearest_even]

    # remove the case in which there are 0 negatives since it is added manually
    all_even = all_even

    # define the items to permute
    items = [-1, 1]

    # generate all permutations
    permutations = list(product(items, repeat=n_edges))

    # keep only the stable situations
    types_of_stable = np.array([p for p in permutations if len([i for i in p if i ==-1]) in all_even])

    return types_of_stable


def simulate(sim_no, n_nodes, N=100, p=.2):

    # Initialize the dictionary to track beliefs
    track = {}

    # Initialize the social network
    G = nx.gnp_random_graph(n=N, p=p)

    # Embed the belief networks
    belief_network_dict = {i:{'belief_network':complete_belief_network(n_nodes=n_nodes, edge_values="default")} for i in range(N)}
    nx.set_node_attributes(G, belief_network_dict)
    # Save the edge_list to use in the simulation
    edge_list = [*belief_network_dict[0]['belief_network'].edges()]

    # Get dictionaries ready for node coloring
    types_of_stable = permute_stable_networks(n_nodes=3)
    types_of_stable_naming, types_of_stable_coloring = analysis_helper.node_coloring_prep(types_of_stable, colors = ["#ef476f","#ffd166","#118ab2","#073b4c"])

    # Simulate
    per_interaction_to_track = 20
    T = N * comb(n_nodes, 2)**2 * per_interaction_to_track
    
    for t in tqdm(range(T+1)):
        
        # Calculate internal energies
        if t % per_interaction_to_track == 0:
            track[t] = {}
            beliefs = np.array([[G.nodes[n]['belief_network'].edges[e]['belief'] for e in edge_list] for n in G.nodes()])
            internal_energies = np.array([internal_energy(G.nodes[n]['belief_network']) for n in G.nodes()])
            track[t]['internal_energies'] = internal_energies
            track[t]['beliefs'] = beliefs

            node_coloring = analysis_helper.node_coloring_on_social_network(G, types_of_stable_naming, types_of_stable_coloring)
            track[t]['node_coloring'] = node_coloring

        if len([*track.keys()]) > n_nodes * N:
            # Stopping criteria if all internal energies = -1 & nothing has changed since the N*(last interaction)
            if all(np.array(internal_energies) == -1):# & (sum(sum(track[[*track.keys()][-1]]['beliefs'] - track[[*track.keys()][-1 * N]]['beliefs'])) == 0):
                break

        if (t % N == 0) & (t > N):
            # this tracks whether an agent's internal energy got better (lower) or not
            track[t]['better_off'] = np.sign(track[t]['internal_energies'] - track[t - N]['internal_energies'])
        
        # Randomly choose a sender, receiver and focal edge
        sender, receiver, focal_edge = choose_sender_receiver_belief(G)
        
        # Calculate the updated weight after agents interact
        b_i_plus_1 = calculate_updated_weight(G, sender, receiver, focal_edge, alpha=1.5, beta=1)

        # Update the belief in the network
        embed_b_i_plus_1_to_belief_network(G, receiver, focal_edge, b_i_plus_1)

    return (sim_no, track), G