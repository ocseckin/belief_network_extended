def simulate(sim_no):

    # Initialize the dictionary to track beliefs
    track = {}

    # Initialize the social network
    N = 500
    p = .2
    seed = 89
    G = nx.gnp_random_graph(n=N, p=p)

    # Embed the belief networks
    n_nodes = 5
    belief_network_dict = {i:{'belief_network':extended_model.complete_belief_network(n_nodes=n_nodes, edge_values="default")} for i in range(N)}
    nx.set_node_attributes(G, belief_network_dict)
    # Save the edge_list to use in the simulation
    edge_list = [*belief_network_dict[0]['belief_network'].edges()]

    # Simulate
    per_interaction_to_track = 20
    T = N * n_nodes * 100

    for t in tqdm(range(T+1)):
        
        # Calculate internal energies
        if t % per_interaction_to_track == 0:
            track[t] = {}
            beliefs = np.array([[G.nodes[n]['belief_network'].edges[e]['belief'] for e in edge_list] for n in G.nodes()])
            internal_energies = np.array([extended_model.internal_energy(G.nodes[n]['belief_network']) for n in G.nodes()])
            track[t]['internal_energies'] = internal_energies
            track[t]['beliefs'] = beliefs

            # Stopping criteria if all internal energies = -1 & nothing has changed from the last interaction
            if t > per_interaction_to_track:

                if all(np.array(internal_energies) == -1) & (sum(sum(track[[*track.keys()][-1]]['beliefs'] - track[[*track.keys()][-2]]['beliefs'])) == 0):
                    break

        if (t % N == 0) & (t > N):
            # this tracks whether an agent's internal energy got better (lower) or not
            track[t]['better_off'] = np.sign(track[t]['internal_energies'] - track[t - N]['internal_energies'])
        
        # Randomly choose a sender, receiver and focal edge
        sender, receiver, focal_edge = extended_model.choose_sender_receiver_belief(G)
        
        # Calculate the updated weight after agents interact
        b_i_plus_1 = extended_model.calculate_updated_weight(G, sender, receiver, focal_edge, alpha=1.5, beta=1)

        # Update the belief in the network
        extended_model.embed_b_i_plus_1_to_belief_network(G, receiver, focal_edge, b_i_plus_1)

    return (sim_no, track)