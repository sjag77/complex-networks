# import pandas as pd
# import community as community_louvain
# import matplotlib.pyplot as plt
# import networkx as nx
# import random

# # File path of the downloaded network
# network_file = "congress.edgelist4"

# # Load the network into NetworkX
# G = nx.read_edgelist(network_file)


# def generate_ws(n, k, p):
#     """Generate a Watts-Strogatz small-world network."""
#     # n: number of nodes, k: each node is connected to k nearest neighbors, p: rewiring probability
#     ws = nx.watts_strogatz_graph(n, k, p)
#     # nx.draw(ws, node_size=50)
#     # plt.show()
#     return ws


# # G = generate_ws(332, 14, 0.1)

# def generate_ba(n, m):
#     """Generate a Barabási-Albert scale-free network."""
#     # n: number of nodes, m: number of edges to attach from a new node to existing nodes
#     ba = nx.barabasi_albert_graph(n, m)
#     # nx.draw(ba, node_size=50)
#     # plt.show()
#     return ba


# # # Example usage
# # G = generate_ba(332, 6)


# def generate_forest_fire(n, fw_prob):
#     """Generate a Forest-Fire network."""
#     # n: number of nodes, fw_prob: forward burning probability
#     ff = nx.gn_graph(n)  # As a simple proxy for Forest Fire model in NetworkX
#     for edge in list(ff.edges()):
#         if random.random() < fw_prob:
#             target = random.choice(list(ff.nodes()))
#             ff.add_edge(edge[1], target)
#     # nx.draw(ff, node_size=50)
#     # plt.show()
#     return ff


# # Example usage
# # G = generate_forest_fire(475, 100010)


# def generate_random_regular_graph(d, n):
#     """
#     Generate a d-regular graph with n nodes.
#     d : Degree of each node
#     n : Number of nodes
#     """
#     # Check for valid parameters
#     if d < 0 or n < 0 or d * n % 2 != 0:
#         raise ValueError(
#             "The degree d and number of nodes n must satisfy the handshake lemma: d*n must be even.")

#     try:
#         # Create a d-regular graph
#         graph = nx.random_regular_graph(d, n)
#         return graph
#     except nx.NetworkXError as e:
#         print("Error in generating the graph:", e)
#         return None


# # Parameters
# degree = 13
# num_nodes = 332

# # Generate the graph
# G = generate_random_regular_graph(degree, num_nodes)


# # Print basic information about the graph
# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())


# # Compute metrics
# average_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
# density = nx.density(G)
# diameter = nx.diameter(G)
# average_clustering_coefficient = nx.average_clustering(G)
# transitivity = nx.transitivity(G)
# average_shortest_path_length = nx.average_shortest_path_length(G)

# # Compute assortativity
# assortativity = nx.degree_assortativity_coefficient(G)

# # Compute top 5 nodes based on centrality measures
# degree_centrality = nx.degree_centrality(G)
# betweenness_centrality = nx.betweenness_centrality(G)
# closeness_centrality = nx.closeness_centrality(G)
# pagerank_centrality = nx.pagerank(G)

# top_5_degree = sorted(degree_centrality.items(),
#                       key=lambda x: x[1], reverse=True)[:5]
# top_5_betweenness = sorted(
#     betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
# top_5_closeness = sorted(closeness_centrality.items(),
#                          key=lambda x: x[1], reverse=True)[:5]
# top_5_pagerank = sorted(pagerank_centrality.items(),
#                         key=lambda x: x[1], reverse=True)[:5]

# # Compute network centralization
# degree_centralization = max(degree_centrality.values()) - \
#     sum(degree_centrality.values())
# betweenness_centralization = max(
#     betweenness_centrality.values()) - sum(betweenness_centrality.values())
# closeness_centralization = max(
#     closeness_centrality.values()) - sum(closeness_centrality.values())
# pagerank_centralization = max(
#     pagerank_centrality.values()) - sum(pagerank_centrality.values())

# network_centralization = {
#     "Degree": degree_centralization,
#     "Betweenness": betweenness_centralization,
#     "Closeness": closeness_centralization,
#     "PageRank": pagerank_centralization
# }

# # Create a DataFrame
# data = {
#     "Metric": ["Average Degree", "Density", "Diameter", "Average Clustering Coefficient",
#                "Transitivity", "Average Shortest Path Length", "Assortativity (Degree Correlation)"],
#     "Value": [average_degree, density, diameter, average_clustering_coefficient,
#               transitivity, average_shortest_path_length, assortativity]
# }

# metrics_df = pd.DataFrame(data)

# # Display the DataFrame
# print("Metrics:")
# print(metrics_df)

# # Display top 5 nodes based on centrality measures
# print("\nTop 5 Nodes based on Centrality Measures:")
# print("Degree Centrality:", top_5_degree)
# print("Betweenness Centrality:", top_5_betweenness)
# print("Closeness Centrality:", top_5_closeness)
# print("PageRank Centrality:", top_5_pagerank)

# # Compute metrics
# # average_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
# # density = nx.density(G)
# # diameter = nx.diameter(G)
# # average_clustering_coefficient = nx.average_clustering(G)
# # transitivity = nx.transitivity(G)
# # average_shortest_path_length = nx.average_shortest_path_length(G)

# # # Print computed metrics
# # print("Average Degree:", average_degree)
# # print("Density:", density)
# # print("Diameter:", diameter)
# # print("Average Clustering Coefficient:", average_clustering_coefficient)
# # print("Transitivity:", transitivity)
# # print("Average Shortest Path Length:", average_shortest_path_length)

# # assortativity = nx.degree_assortativity_coefficient(G)
# # print("Assortativity (Degree Correlation):", assortativity)

# # # Degree centrality
# # degree_centrality = nx.degree_centrality(G)
# # top_5_degree = sorted(degree_centrality.items(),
# #                       key=lambda x: x[1], reverse=True)[:5]
# # print("Top 5 nodes based on Degree Centrality:")
# # print(top_5_degree)

# # # Betweenness centrality
# # betweenness_centrality = nx.betweenness_centrality(G)
# # top_5_betweenness = sorted(
# #     betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
# # print("Top 5 nodes based on Betweenness Centrality:")
# # print(top_5_betweenness)

# # # Closeness centrality
# # closeness_centrality = nx.closeness_centrality(G)
# # top_5_closeness = sorted(closeness_centrality.items(),
# #                          key=lambda x: x[1], reverse=True)[:5]
# # print("Top 5 nodes based on Closeness Centrality:")
# # print(top_5_closeness)

# # # PageRank centrality
# # pagerank_centrality = nx.pagerank(G)
# # top_5_pagerank = sorted(pagerank_centrality.items(),
# #                         key=lambda x: x[1], reverse=True)[:5]
# # print("Top 5 nodes based on PageRank Centrality:")
# # print(top_5_pagerank)


# def calculate_degree_centralization(G):
#     degree_centrality = nx.degree_centrality(G)
#     max_degree_centrality = max(degree_centrality.values())
#     sum_max_diff = sum(max_degree_centrality -
#                        dc for dc in degree_centrality.values())
#     max_possible_sum = (len(G) - 1) * (len(G) - 1)
#     centralization = sum_max_diff / max_possible_sum
#     return centralization


# # Example usage:

# degree_centralization = calculate_degree_centralization(G)
# print("Degree Centralization:", degree_centralization)


# def calculate_closeness_centralization(G):
#     closeness_centrality = nx.closeness_centrality(G)
#     max_closeness_centrality = max(closeness_centrality.values())
#     sum_max_diff = sum(max_closeness_centrality -
#                        cc for cc in closeness_centrality.values())
#     max_possible_sum = (len(G) - 1) * (1 - min(closeness_centrality.values()))
#     centralization = sum_max_diff / max_possible_sum
#     return centralization


# # Example usage:
# closeness_centralization = calculate_closeness_centralization(G)
# print("Closeness Centralization:", closeness_centralization)


# def calculate_betweenness_centralization(G):
#     betweenness_centrality = nx.betweenness_centrality(G)
#     max_betweenness_centrality = max(betweenness_centrality.values())
#     sum_max_diff = sum(max_betweenness_centrality -
#                        bc for bc in betweenness_centrality.values())
#     max_possible_sum = (len(G) - 1) * (len(G) - 2) / 2
#     centralization = sum_max_diff / max_possible_sum
#     return centralization


# # Example usage:
# betweenness_centralization = calculate_betweenness_centralization(G)
# print("Betweenness Centralization:", betweenness_centralization)


# def calculate_pagerank_centralization(G):
#     pagerank = nx.pagerank(G)
#     max_pagerank = max(pagerank.values())
#     sum_max_diff = sum(max_pagerank - pr for pr in pagerank.values())
#     max_possible_sum = (len(G) - 1) * (1 - min(pagerank.values()))
#     centralization = sum_max_diff / max_possible_sum
#     return centralization


# pagerank_centralization = calculate_pagerank_centralization(G)
# print("Pagerank Centralization:", pagerank_centralization)

# # Community detection using Louvain method
# partition = community_louvain.best_partition(G)
# modularity = community_louvain.modularity(partition, G)

# print("\nModularity of the detected communities:", modularity)

# # Visualize the graph with communities
# plt.figure(figsize=(12, 10))
# pos = nx.fruchterman_reingold_layout(G)
# cmap = plt.get_cmap('viridis')
# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
#                        node_color=list(partition.values()), cmap=cmap)
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.title('Graph with Louvain Communities')
# plt.show()

# # Analysis and Interpretation of the Results
# num_communities = len(set(partition.values()))
# print("\nNumber of communities detected:", num_communities)

# # Distribution of community sizes
# community_sizes = pd.Series(list(partition.values())).value_counts()
# print("\nCommunity sizes:\n", community_sizes)

# # Visualize the complete graph with Fruchterman-Reingold layout
# plt.figure(figsize=(12, 10))  # Increase the figure size
# pos = nx.fruchterman_reingold_layout(G)
# nx.draw(G, pos, node_color='skyblue', node_size=20, with_labels=False)
# plt.title('Improved Complete Graph Visualization')
# plt.show()


# # network_centralization = {
# #     "Degree": max(degree_centralization.values()) - sum(degree_centralization.values()),
# #     "Betweenness": max(betweenness_centralization.values()) - sum(betweenness_centralization.values()),
# #     "Closeness": max(closeness_centralization.values()) - sum(closeness_centralization.values()),
# #     "PageRank": max(pagerank_centralization.values()) - sum(pagerank_centralization.values())
# # }
# # print("Network Centralization:")
# # print(network_centralization)


# # Compute degree distribution
# degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
# degree_count = nx.degree_histogram(G)

# # # Plot degree distribution
# # plt.bar(range(len(degree_count)), degree_count, width=0.8, color='r')
# # plt.xlabel('Degree')
# # plt.ylabel('Frequency')
# # plt.title('Degree Distribution')
# # plt.show()
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load the dataset
dataset_file = 'congress.edgelist'

# Function to parse the edge list
def parse_edgelist(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 2:
                node1 = int(parts[0])
                node2 = int(parts[1])
                edges.append((node1, node2))
    return edges

# Parse the edges
edges = parse_edgelist(dataset_file)

# Create the graph
G2 = nx.Graph()
G2.add_edges_from(edges)

# Ensure graph is not empty
if len(G2.nodes) == 0:
    print("The graph is empty. Please check the input file and parsing function.")
else:
    # Visualize the filtered graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G2)
    nx.draw(G2, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=12, font_color='black')
    plt.title('Filtered Network Graph')
    plt.show()

    # Define the GraphRNN Model
    class GraphRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(GraphRNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x, hidden):
            out, hidden = self.rnn(x, hidden)
            out = self.fc(out)
            return out, hidden

        def init_hidden(self, batch_size):
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    # Prepare the data for training
    def prepare_data(G):
        adj_matrix = nx.adjacency_matrix(G).todense()
        adj_matrix = np.array(adj_matrix)
        return torch.tensor(adj_matrix, dtype=torch.float32)

    train_data = prepare_data(G2)

    # Check the shape of train_data before reshaping
    print("Original shape of train_data:", train_data.shape)

    # Ensure train_data is in correct shape
    batch_size = 1
    num_nodes = train_data.shape[0]
    sequence_length = num_nodes  # Each node's adjacency list is a sequence
    input_size = num_nodes  # The input size is the number of nodes

    # Reshape train_data to match the GRU input requirements: [batch_size, sequence_length, input_size]
    train_data = train_data.unsqueeze(0)  # Add batch dimension
    train_data = train_data.permute(0, 2, 1)  # Permute to match [batch_size, sequence_length, input_size]

    # Debug print to check shape
    print("Shape of train_data after reshaping:", train_data.shape)  # Should be [1, num_nodes, num_nodes]

    # Hyperparameters
    hidden_size = 64  # Increased hidden size
    num_layers = 2  # Added more layers
    output_size = num_nodes  # Set output_size to the number of nodes
    num_epochs = 1000  # Increased number of epochs
    learning_rate = 0.0005  # Decreased learning rate for finer adjustments

    # Model, Loss, Optimizer
    model = GraphRNN(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the Model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        hidden = model.init_hidden(batch_size)
        
        input_data = train_data  # Use reshaped train_data
        target_data = train_data  # Ensure target data shape matches input data
        
        output, hidden = model(input_data, hidden)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Generate a new graph
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(batch_size)
        input_data = train_data
        generated_data, hidden = model(input_data, hidden)

    # Convert the generated data back to a graph
    generated_adj_matrix = generated_data.squeeze(0).numpy()
    generated_graph = nx.from_numpy_array(generated_adj_matrix)

    # Ensure the generated graph is not empty
    if len(generated_graph.nodes) == 0:
        print("The generated graph is empty. Please check the model output.")
    else:
        # Visualize the generated graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(generated_graph)
        nx.draw(generated_graph, pos, with_labels=True, node_size=500, node_color='orange', font_size=12, font_color='black')
        plt.title('Generated Network Graph')
        plt.show()

        # Compute and print metrics
        try:
            print("Number of nodes:", generated_graph.number_of_nodes())
            print("Number of edges:", generated_graph.number_of_edges())

            # Compute metrics
            average_degree = sum(dict(generated_graph.degree()).values()) / generated_graph.number_of_nodes()
            density = nx.density(generated_graph)
            diameter = nx.diameter(generated_graph) if nx.is_connected(generated_graph) else "Graph is not connected"
            average_clustering_coefficient = nx.average_clustering(generated_graph)
            transitivity = nx.transitivity(generated_graph)
            average_shortest_path_length = nx.average_shortest_path_length(generated_graph) if nx.is_connected(generated_graph) else "Graph is not connected"

            # Compute assortativity
            assortativity = nx.degree_assortativity_coefficient(generated_graph)

            # Print metrics
            print(f"Average Degree: {average_degree}")
            print(f"Density: {density}")
            print(f"Diameter: {diameter}")
            print(f"Average Clustering Coefficient: {average_clustering_coefficient}")
            print(f"Transitivity: {transitivity}")
            print(f"Average Shortest Path Length: {average_shortest_path_length}")
            print(f"Assortativity (Degree Correlation): {assortativity}")

        except Exception as e:
            print(f"An error occurred while computing metrics: {e}")
