import torch
from torch_geometric.nn import radius_graph
from tqdm import tqdm
from torch_geometric.data import Data

# Function to compute the number of edges in a graph
def compute_num_edges(data: Data, radius=8.0):
    """
    Compute the number of edges in a graph using a radius graph.
    """
    edge_index = radius_graph(data.pos, r=radius, loop=True)
    return edge_index.size(1)

# Function to filter the largest graphs
def filter_largest_graphs(dataset, top_k=10, radius=8.0):
    """
    Filters the dataset to retain only the top_k largest graphs by number of edges.
    """
    graph_edge_counts = []

    # Compute the number of edges for each graph
    for i, data in enumerate(tqdm(dataset, desc="Computing edge counts")):
        num_edges = compute_num_edges(data, radius=radius)
        graph_edge_counts.append((i, num_edges))

    # Sort by number of edges in descending order
    sorted_graphs = sorted(graph_edge_counts, key=lambda x: x[1], reverse=True)

    # Select the top_k largest graphs
    largest_graph_indices = [idx for idx, _ in sorted_graphs[:top_k]]

    # Filter the dataset
    largest_graphs = [dataset[idx] for idx in largest_graph_indices]

    print(f"Selected {len(largest_graphs)} largest graphs with edge counts: {[count for _, count in sorted_graphs[:top_k]]}")
    return largest_graphs

# Load the original dataset
train_data = torch.load('./datasets/processed_train.pt')

# Filter the dataset to keep only the 10 largest graphs
largest_graphs = filter_largest_graphs(train_data, top_k=10, radius=8.0)

# Save the filtered dataset
torch.save(largest_graphs, './datasets/largest_graphs.pt')
print("Filtered dataset with largest graphs saved successfully.")
