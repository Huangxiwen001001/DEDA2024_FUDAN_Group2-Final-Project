from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import random
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, classification_report
import re
from datetime import datetime

# data process
def adjacency_matrix_to_PyG(adj_matrix, y):
    """
    Convert an undirected graph adjacency matrix into an edge index matrix of shape [2, num_edges].

    :param adj_matrix: Adjacency matrix, a 2D array where adj_matrix[i][j] represents the weight of the edge from node i to node j.
    :param y: Target labels.
    :return: A tensor with shape [2, num_edges] containing edge indices.
    """
    edge_index = []
    edge_attr = []
    x = []
    num_nodes = adj_matrix.shape[0]  # Get the number of nodes
    for i in range(num_nodes):
        x.append([i])
        for j in range(num_nodes):
            if adj_matrix[i][j] != 0 and i != j:  # Exclude self-loops and non-connections
                edge_index.append([i, j])
                edge_attr.append([adj_matrix[i][j]])

    # Convert the list of edges to a tensor of shape [2, num_edges]
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

    x_tensor = torch.tensor(x, dtype=torch.float)

    y_tensor = torch.tensor(y, dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor, edge_attr=edge_attr_tensor)
    return data

# Extract date from filename
def extract_date_from_filename(filename):
    date_str = re.search(r'_(\d{8})_', filename)
    if date_str:
        return date_str.group(1)
    return None

folder_path = r"3_ML\data_networks_csv"

# List all files in the directory
all_files = os.listdir(folder_path)

# Check files that match the pattern and extract dates
file_names = [f for f in all_files if f.startswith('stock_vol_bad_') and f.endswith('非线性网络.csv')]
file_dates = [extract_date_from_filename(f) for f in file_names if extract_date_from_filename(f) is not None]

# Convert to date objects for comparison
file_dates = pd.to_datetime(file_dates, format='%Y%m%d').strftime('%Y%m%d').tolist()

# Compute missing dates
dates = sorted(set(file_dates))  # Remove duplicates and sort
missing_dates = [date for date in dates if date not in file_dates]

# Output missing dates
print(f"Number of missing dates: {len(missing_dates)}, they are: {missing_dates[:2]}")  # List only the first two missing dates as an example

# Initialize an empty list to store adjacency matrix data
adjacency_matrices = []
node_features = []
data_list = []
# Define event periods
event_periods = [
    (datetime(2007, 7, 26), datetime(2009, 12, 31)),  # Global financial crisis
    (datetime(2013, 6, 7), datetime(2013, 12, 31)),  # Bank liquidity crisis
    (datetime(2015, 6, 15), datetime(2016, 12, 20))  # Stock market crash
]

# Check the number of trading days during each event period
for i, period in enumerate(event_periods):
    print(f"Event {i+1} trading days: {len(pd.date_range(start=period[0], end=period[1], freq='B'))}")

# Combine event period dates
event_dates = set()
for start_date, end_date in event_periods:
    period_dates = pd.date_range(start=start_date, end=end_date, freq='B').strftime("%Y%m%d").tolist()
    event_dates.update(period_dates)

# Initialize target list
target_list = [1 if date in event_dates else 0 for date in dates]

# Output check
print(f"Number of days with target=1 (within event periods): {sum(target_list)}")

# Format target list
formatted_target_list = [str(date) for date in target_list]

# Find missing dates
missing_dates = [date for date in formatted_target_list if not any(date in f for f in file_names)]

# Output missing dates
print(f"Number of missing dates: {len(missing_dates)}, they are: {missing_dates}")

# Load adjacency matrices from CSV files
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    adjacency_matrix = pd.read_csv(file_path, skiprows=1).values
    adjacency_matrices.append(adjacency_matrix)
print(f"Loaded adjacency matrices: {len(adjacency_matrices)}")
print(f"target list: {len(target_list)}")
print(len(adjacency_matrices), len(target_list))

# Get the date part from the file names and find missing dates
loaded_dates = [file_name.split('_')[3] for file_name in file_names]
missing_dates = [date for date in dates if date.replace('-', '') not in loaded_dates]

print(f"Missing files for dates: {missing_dates}")

# Convert adjacency matrices to PyG data
for i in range(len(adjacency_matrices)):
    data = adjacency_matrix_to_PyG(adjacency_matrices[i], target_list[i])
    data_list.append(data)
print("Data list length:", len(data_list))

# Update: format dates
formatted_dates = [date.replace('-', '') for date in dates]

# Update: check missing files
missing_dates = [date for date in formatted_dates if date not in loaded_dates]
print(f"Missing dates: {len(missing_dates)}, they are: {missing_dates[:2]}")

# Convert dates to datetime objects
date_objects = [datetime.strptime(date, "%Y%m%d") for date in dates]

# Define training and testing date ranges
train_val_start_date = datetime(2008, 5, 30)
train_val_end_date = datetime(2014, 7, 31)
test_start_date = datetime(2014, 8, 1)
test_end_date = datetime(2022, 12, 31)

# Split data into training, validation, and testing sets based on date ranges
train_val_indices = [i for i, date in enumerate(date_objects) if train_val_start_date <= date <= train_val_end_date]
test_indices = [i for i, date in enumerate(date_objects) if test_start_date <= date <= test_end_date]

# Print dataset split
print(f"Train/Validation indices: {len(train_val_indices)}")
print(f"Test indices: {len(test_indices)}")

# Shuffle training/validation indices
random.shuffle(train_val_indices)

# Determine the number of training samples (75% of 2008-2018 data)
num_train = int(0.75 * len(train_val_indices))

# Split into training and validation sets
train_indices = train_val_indices[:num_train]
val_indices = train_val_indices[num_train:]

# Create data loaders
train_dataset = Batch.from_data_list([data_list[i] for i in train_indices])
val_dataset = Batch.from_data_list([data_list[i] for i in val_indices])
test_dataset = Batch.from_data_list([data_list[i] for i in test_indices])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes=2, num_node_features=1, num_edge_features=1):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels, num_edge_features)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, num_edge_features)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, num_edge_features)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.lin(x)

        return x

model = GCN(hidden_channels=65)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

def validate(val_loader):
    model.eval()

    correct = 0
    for data in val_loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(val_loader.dataset)

def tes(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate over the batches in the test set.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Forward pass
        pred = out.argmax(dim=1)  # Use the class with the highest probability
        correct += int((pred == data.y).sum())  # Check with the true label
    return correct / len(loader.dataset)

# Evaluation function
def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels

def evaluation_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate F-Alarm (false alarm rate)
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + np.diag(cm) + cm.sum(axis=1) - np.diag(cm))
    F_Alarm = FP / (FP + TN + 1e-6)
    F_Alarm_mean = np.mean(F_Alarm)

    metric = {
        'Model': model_name,
        'Accuracy': accuracy,
        'F1-score': f1,
        'F-Alarm': F_Alarm_mean
    }
    print(metric)
    return metric

# Record validation set metrics
val_metrics = []

# Initialize lists to store accuracy for each epoch
train_acc_list = []
val_acc_list = []
test_acc_list = []

for epoch in range(1, 351):
    train()
    train_acc = tes(train_loader)
    test_acc = tes(test_loader)
    val_acc = tes(val_loader)
    # Record accuracies
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    test_acc_list.append(test_acc)
    val_preds, val_labels = evaluate(val_loader)
    val_metrics.append(evaluation_metrics(val_labels, val_preds, 'Validate'))
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Validate Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(1)

# Save as CSV file
df_val = pd.DataFrame(val_metrics)
df_val.to_csv(r'3_ML\accuraciesGGNN.csv', index=False)
print("File saved successfully.")


# Record test set evaluation results
test_preds, test_labels = evaluate(test_loader)

print(np.unique(test_preds))

# Set font properties for Chinese characters
rcParams['font.family'] = 'SimHei'  # SimHei is a font supporting Chinese
rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Plot heatmap based on test set results
plt.figure(figsize=(12, 6))

# Set up the heatmap with adjusted color intensity for deeper blue
sns.heatmap(
    np.array(test_preds).reshape(1, -1),
    cmap=sns.color_palette("Blues", as_cmap=True),  # Use a deeper blue colormap
    cbar=False,
    yticklabels=[''],
    mask=(np.array(test_preds).reshape(1, -1) == 0),  # Mask transparent areas where the value is zero
    vmin=np.percentile(test_preds, 10),  # Set minimum value to make lower values darker
    vmax=np.percentile(test_preds, 95),  # Cap maximum value for higher contrast
)

# Modify x-ticks to show years 2008 to 2022, with labels every two years
years = np.arange(2008, 2023, 2)  # Create an array of years from 2008 to 2022 with a step of 2
num_ticks = len(test_preds)  # Total number of ticks based on test_preds length
tick_positions = np.linspace(2, num_ticks - 1, len(years))  # Generate positions evenly spaced across the x-axis
plt.xticks(tick_positions, years, rotation=0, ha="right")  # Set x-tick positions and labels
plt.title('Prediction Results (Test Set)')

# Save heatmap to local file with transparent background
plt.savefig(
    r'3_ML\graph_prediction_results.png',
    dpi=300,
    bbox_inches='tight',
    transparent=True
)

plt.show()


# Plot accuracy curves for train, validation, and test sets
plt.figure(figsize=(12, 6))
epochs = list(range(1, 351))  # Create a list for X-axis (1 to 350)
plt.plot(epochs, train_acc_list, label='Train Accuracy')
plt.plot(epochs, val_acc_list, label='Validation Accuracy')
plt.plot(epochs, test_acc_list, label='Test Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.grid(True)

# Save the accuracy curve to a jpg file
plt.savefig(r'3_ML\graph_accuracy_curve.png', dpi=300, bbox_inches='tight', transparent=True)

plt.show()
