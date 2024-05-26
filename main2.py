# Load the dataset
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

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

# Visualize the dataset graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G2)
nx.draw(G2, pos, with_labels=True, node_size=500,
        node_color='skyblue', font_size=12, font_color='black')
plt.title('Dataset Network Graph')
plt.show()

# Compute and print dataset graph metrics
print("Number of nodes in dataset graph:", G2.number_of_nodes())
print("Number of edges in dataset graph:", G2.number_of_edges())

# Compute metrics
average_degree = sum(dict(G2.degree()).values()) / G2.number_of_nodes()
density = nx.density(G2)
diameter = nx.diameter(G2) if nx.is_connected(G2) else "Graph is not connected"
average_clustering_coefficient = nx.average_clustering(G2)
transitivity = nx.transitivity(G2)
average_shortest_path_length = nx.average_shortest_path_length(
    G2) if nx.is_connected(G2) else "Graph is not connected"

# Print metrics
metrics_data = {
    "Metric": ["Average Degree", "Density", "Diameter", "Average Clustering Coefficient", "Transitivity", "Average Shortest Path Length"],
    "Value": [average_degree, density, diameter, average_clustering_coefficient, transitivity, average_shortest_path_length]
}

metrics_df = pd.DataFrame(metrics_data)
print("Metrics for Dataset Graph:")
print(metrics_df)


def graph_to_adj_matrix(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    adj_matrix = np.array(adj_matrix, dtype=np.float32)
    adj_matrix /= adj_matrix.max()  # Normalize the adjacency matrix
    return adj_matrix


adj_matrix = graph_to_adj_matrix(G2)


def prepare_geometric_data(adj_matrix):
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
    x = torch.tensor(np.eye(adj_matrix.shape[0]), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


data = prepare_geometric_data(adj_matrix)


def custom_loss(generated_graph, target_metrics):
    try:
        if nx.is_empty(generated_graph):
            # High loss if the graph is empty
            return torch.tensor(float('inf'), requires_grad=True)

        average_degree = sum(dict(generated_graph.degree()
                                  ).values()) / generated_graph.number_of_nodes()
        density = nx.density(generated_graph)
        diameter = nx.diameter(generated_graph) if nx.is_connected(
            generated_graph) else 0
        average_clustering_coefficient = nx.average_clustering(generated_graph)
        transitivity = nx.transitivity(generated_graph)
        average_shortest_path_length = nx.average_shortest_path_length(
            generated_graph) if nx.is_connected(generated_graph) else 0

        # Penalize high density and other unrealistic metrics
        density_penalty = max(0, density - 0.1)
        clustering_penalty = max(0, average_clustering_coefficient - 0.4)
        transitivity_penalty = max(0, transitivity - 0.4)
        short_path_penalty = max(0, 1.0 - average_shortest_path_length)

        generated_metrics = torch.tensor([
            average_degree,
            density,
            average_clustering_coefficient,
            transitivity,
            average_shortest_path_length
        ], dtype=torch.float32, requires_grad=True)

        loss = nn.MSELoss()(generated_metrics, target_metrics) + density_penalty + \
            clustering_penalty + transitivity_penalty + short_path_penalty
    except Exception as e:
        # Return a large loss value if any error occurs
        loss = torch.tensor(float('inf'), requires_grad=True)

    return loss


class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


# Hyperparameters
input_dim = data.num_features
hidden_dim = 32  # Reduced hidden size for simplicity
output_dim = data.num_features
num_epochs = 100
learning_rate = 0.01
dropout_rate = 0.5

# Model, Loss, Optimizer
model = SimpleGCN(input_dim, hidden_dim, output_dim, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Prepare the data for training
train_data = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(
    0)  # Add batch dimension

# Training the Model
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data.x, data.edge_index)

    # Compute loss
    loss = criterion(output, train_data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def apply_dynamic_threshold(adj_matrix, target_num_edges):
    """ Apply a dynamic threshold to get the desired number of edges. """
    flattened = adj_matrix.flatten()
    sorted_values = np.sort(flattened)[::-1]
    threshold_index = target_num_edges if target_num_edges < len(
        sorted_values) else -1
    threshold_value = sorted_values[threshold_index]
    binary_adj_matrix = (adj_matrix >= threshold_value).astype(int)
    return binary_adj_matrix


def ensure_connected_graph(binary_adj_matrix):
    """ Ensure the graph is connected by adding edges if necessary. """
    G = nx.from_numpy_array(binary_adj_matrix)
    if nx.is_connected(G):
        return binary_adj_matrix
    else:
        components = list(nx.connected_components(G))
        while len(components) > 1:
            component1 = components.pop()
            component2 = components.pop()
            node1 = list(component1)[0]
            node2 = list(component2)[0]
            binary_adj_matrix[node1, node2] = 1
            binary_adj_matrix[node2, node1] = 1
            G = nx.from_numpy_array(binary_adj_matrix)
            components = list(nx.connected_components(G))
        return binary_adj_matrix


# Generate a new graph
model.eval()
with torch.no_grad():
    generated_adj_matrix = model(data.x, data.edge_index).numpy()
    np.fill_diagonal(generated_adj_matrix, 0)  # Remove self-loops
    binary_adj_matrix = apply_dynamic_threshold(
        generated_adj_matrix, target_num_edges=10222)
    connected_adj_matrix = ensure_connected_graph(binary_adj_matrix)
    generated_graph = nx.from_numpy_array(connected_adj_matrix)

# Ensure the generated graph is not empty
if len(generated_graph.nodes) == 0:
    print("The generated graph is empty. Please check the model output.")
else:
    # Visualize the generated graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(generated_graph)
    nx.draw(generated_graph, pos, with_labels=True, node_size=500,
            node_color='orange', font_size=12, font_color='black')
    plt.title('Generated Network Graph')
    plt.show()

    # Compute and print metrics
    try:
        print("Number of nodes:", generated_graph.number_of_nodes())
        print("Number of edges:", generated_graph.number_of_edges())

        # Compute metrics
        average_degree = sum(dict(generated_graph.degree()
                                ).values()) / generated_graph.number_of_nodes()
        density = nx.density(generated_graph)
        diameter = nx.diameter(generated_graph) if nx.is_connected(
            generated_graph) else "Graph is not connected"
        average_clustering_coefficient = nx.average_clustering(generated_graph)
        transitivity = nx.transitivity(generated_graph)
        average_shortest_path_length = nx.average_shortest_path_length(
            generated_graph) if nx.is_connected(generated_graph) else "Graph is not connected"

        print("Generated Graph Metrics:")
        print(f"Average Degree: {average_degree}")
        print(f"Density: {density}")
        print(f"Diameter: {diameter}")
        print(f"Average Clustering Coefficient: {
              average_clustering_coefficient}")
        print(f"Transitivity: {transitivity}")
        print(f"Average Shortest Path Length: {average_shortest_path_length}")

    except Exception as e:
        print(f"An error occurred while computing metrics: {e}")
