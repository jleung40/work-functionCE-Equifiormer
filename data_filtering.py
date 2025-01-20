import torch
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from tqdm import tqdm

# Define the memory estimation function
def estimate_memory_usage(data: Data, r=8):
    """
    Estimate memory usage of a PyG Data object.
    Returns memory in MB.
    """
    try:
        num_nodes = data.x.size(0)
        edge_index = radius_graph(data.pos, r=r, loop=True)  # Generate edge_index dynamically
        num_edges = edge_index.size(1)
        node_feature_size = data.x.size(1) * 4  # Assuming float32 (4 bytes)
        edge_feature_size = 4  # Adjust if edge attributes are included
        other_tensors = (
            data.y.size(0) * 4 +
            data.cell.size(0) * data.cell.size(1) * 4
        )

        memory_bytes = (
            num_nodes * node_feature_size +
            num_edges * edge_feature_size +
            other_tensors
        )
        return memory_bytes / (1024 ** 2)  # Convert to MB
    except Exception as e:
        print(f"Error estimating memory for data point: {e}")
        return float('inf')  # Skip the problematic graph

# Filtering function
def filter_large_graphs(dataset, max_memory_mb=4000, r=8):
    """
    Filters out graphs with estimated memory usage > max_memory_mb.
    """
    filtered_data = []
    skipped_count = 0

    for data in tqdm(dataset, desc="Filtering dataset"):
        try:
            # Dynamically generate edge_index for memory estimation
            data.edge_index = radius_graph(data.pos, r=r, loop=True)
            memory_mb = estimate_memory_usage(data, r=r)

            # Filter based on memory
            if memory_mb <= max_memory_mb:
                # Remove edge_index before saving
                del data.edge_index
                filtered_data.append(data)
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error processing data point: {e}")
            skipped_count += 1

    print(f"Filtered dataset size: {len(filtered_data)} graphs")
    print(f"Skipped {skipped_count} graphs exceeding {max_memory_mb} MB or causing errors")
    return filtered_data

# Load the original dataset
train_data = torch.load('./datasets/processed_train.pt')
val_data = torch.load('./datasets/processed_val.pt')
test_data = torch.load('./datasets/processed_test.pt')

# Apply filtering
max_memory_mb = 4000  # Adjust this value as needed
radius_value = 9.0  # Smaller radius value to reduce edge density
filtered_train_data = filter_large_graphs(train_data, max_memory_mb=max_memory_mb, r=radius_value)
filtered_val_data = filter_large_graphs(val_data, max_memory_mb=max_memory_mb, r=radius_value)
filtered_test_data = filter_large_graphs(test_data, max_memory_mb=max_memory_mb, r=radius_value)

# Save the filtered datasets
torch.save(filtered_train_data, './datasets/filtered_train.pt')
torch.save(filtered_val_data, './datasets/filtered_val.pt')
torch.save(filtered_test_data, './datasets/filtered_test.pt')

print("Filtered datasets saved successfully.")
