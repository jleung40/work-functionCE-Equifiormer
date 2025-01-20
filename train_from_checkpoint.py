import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Load Preprocessed Data
train_data = torch.load('./datasets/processed_train.pt')
val_data = torch.load('./datasets/processed_val.pt')
test_data = torch.load('./datasets/processed_test.pt')

# Add 'natoms' attribute to data samples
for dataset in [train_data, val_data, test_data]:
    for data in dataset:
        data.natoms = torch.tensor([data.x.shape[0]])

print(f"Data keys: {train_data[0].keys()}")
print(f"Number of atoms: {train_data[0].natoms.item()}")

# Define the custom collate function
def custom_collate_fn(batch):
    batch = Batch.from_data_list(batch)
    batch.cell = torch.stack([data.cell for data in batch.to_data_list()], dim=0)
    return batch

# Create DataLoaders
train_loader = DataLoader(
    train_data,
    batch_size=2,
    shuffle=True,
    collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    val_data,
    batch_size=2,
    collate_fn=custom_collate_fn
)
test_loader = DataLoader(
    test_data,
    batch_size=2,
    collate_fn=custom_collate_fn
)

# Load Config File
config_path = './configs/equiformer_v2_N@12_L@6_M@2_epochs@30.yml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EquiformerV2_OC20(
    num_atoms=None,
    bond_feat_dim=None,
    num_targets=1,
    max_radius=config['model']['max_radius'],
    regress_forces=False,
    num_layers=config['model']['num_layers'],
    sphere_channels=config['model']['sphere_channels'],
    attn_hidden_channels=config['model']['attn_hidden_channels'],
    num_heads=config['model']['num_heads'],
    attn_alpha_channels=config['model']['attn_alpha_channels'],
    attn_value_channels=config['model']['attn_value_channels'],
    ffn_hidden_channels=config['model']['ffn_hidden_channels'],
    norm_type=config['model']['norm_type'],
    lmax_list=config['model']['lmax_list'],
    mmax_list=config['model']['mmax_list'],
    edge_channels=config['model']['edge_channels'],
    alpha_drop=config['model']['alpha_drop'],
    drop_path_rate=config['model']['drop_path_rate'],
    proj_drop=config['model']['proj_drop']
).to(device)

# Load Checkpoint
checkpoint_path = "./checkpoints/best_model.pt"
start_epoch = 1
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

if torch.cuda.is_available() and torch.cuda.current_device():
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming training from checkpoint...")
    except FileNotFoundError:
        print("Checkpoint not found. Starting fresh.")

# Loss function
loss_fn = torch.nn.MSELoss()

# Metric: Mean Squared Error
def calculate_mse(predictions, targets):
    """
    Compute Mean Squared Error (MSE) between predictions and targets.
    """
    mse = torch.mean((predictions - targets) ** 2)
    return mse.item()

# Training Loop
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        batch = batch.to(device)

        # Mixed precision forward pass
        with autocast():
            outputs = model(batch).view(-1)
            target = batch.y.to(device).view(-1)
            loss = loss_fn(outputs, target) / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Training Loss = {avg_loss:.4f}")
    return avg_loss

# Validation Loop
def validate():
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch).view(-1)
            target = batch.y.to(device).view(-1)
            all_predictions.append(outputs)
            all_targets.append(target)
            loss = loss_fn(outputs, target)
            total_loss += loss.item()

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    mse = calculate_mse(all_predictions, all_targets)

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss = {avg_loss:.4f}, MSE = {mse:.4f}")
    return avg_loss, mse

# Train and Validate
num_epochs = 30
best_val_loss = float('inf')

for epoch in range(start_epoch, num_epochs + 1):
    train_loss = train_one_epoch(epoch)
    val_loss, val_mse = validate()

    # Save checkpoint if validation improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch
        }, './checkpoints/best_model.pt')
        print(f"Saved best model at epoch {epoch} with validation loss {best_val_loss:.4f}")

# Evaluate on Test Set
def test():
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch).view(-1)
            target = batch.y.to(device).view(-1)
            all_predictions.append(outputs)
            all_targets.append(target)
            loss = loss_fn(outputs, target)
            total_loss += loss.item()

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    mse = calculate_mse(all_predictions, all_targets)

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss = {avg_loss:.4f}, MSE = {mse:.4f}")
    return avg_loss, mse

print("Evaluating on test set...")
test_loss, test_mse = test()
