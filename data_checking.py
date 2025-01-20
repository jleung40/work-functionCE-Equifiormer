import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

# Paths to your preprocessed datasets
train_data_path = './datasets/processed_train.pt'
val_data_path = './datasets/processed_val.pt'
test_data_path = './datasets/processed_test.pt'

# Load the data from disk
train_data = torch.load(train_data_path)
val_data = torch.load(val_data_path)
test_data = torch.load(test_data_path)

# Verify that the data is a list of Data objects
print(f"Type of train_data: {type(train_data)}")
print(f"Type of first item in train_data: {type(train_data[0])}")

# Adjust 'y' to have the correct shape if necessary
for dataset in [train_data, val_data, test_data]:
    for data in dataset:
        if data.y.shape[0] > 1:
            data.y = data.y.mean(dim=0, keepdim=True)

# Define the custom collate function
def custom_collate_fn(data_list):
    print("Custom collate ran")
    # Extract 'cell' tensors
    cells = [data.cell for data in data_list]
    
    # Debug: Print the shapes of individual 'cell' tensors
    print(f"Number of cells: {len(cells)}")
    for i, cell in enumerate(cells):
        print(f"Cell {i} shape before reshape: {cell.shape}")
        if cell.numel() == 9 and cell.shape != (3, 3):
            cell = cell.view(3, 3)
            cells[i] = cell  # Update the cell in the list
        print(f"Cell {i} shape after reshape: {cell.shape}")
    
    # Remove 'cell' from data objects
    for data in data_list:
        del data.cell
    
    # Batch the data
    batch = Batch.from_data_list(data_list)
    
    # Stack 'cell' tensors along the batch dimension
    batch.cell = torch.stack(cells, dim=0)
    
    # Debug: Print the shape of batch.cell
    print(f"Batch cell shape after stacking: {batch.cell.shape}")
    
    return batch

# Create DataLoader instances with the custom collate function
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    val_data,
    batch_size=32,
    collate_fn=custom_collate_fn
)
test_loader = DataLoader(
    test_data,
    batch_size=32,
    collate_fn=custom_collate_fn
)

print(f"Data cell shape: {train_data[0].cell.shape}")  # Should be torch.Size([3, 3])

print("Starting DataLoader iteration...")
# Iterate through the DataLoader
for idx, data in enumerate(train_loader):
    print(f"Batch {idx + 1}:")
    print(f"  Input (x): {data.x.shape}")
    print(f"  Positions (pos): {data.pos.shape}")
    if data.edge_index is not None:
        print(f"  Edge Index (edge_index): {data.edge_index.shape}")
    else:
        print(f"  Edge Index (edge_index): None")
    print(f"  Target (y): {data.y.shape}")
    print(f"  Batch vector: {data.batch.shape}")
    print(f"  Number of graphs in batch: {data.num_graphs}")
    print(f"  Cell shape: {data.cell.shape}")  # Should be [batch_size, 3, 3]
    # Debugging edge_index content
    if data.edge_index is not None:
        print(f"  Edge Index values (sample): {data.edge_index[:, :10]}")  # Print first 10 edges for inspection
    if idx == 4:
        break
