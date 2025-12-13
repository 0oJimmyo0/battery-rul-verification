import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load CSV (upload first)
df = pd.read_csv("/Users/wuwenrong/Desktop/Battery_dataset.csv")

# Drop categorical column
df = df.drop(columns=["battery_id"])

# Select target
target_col = "RUL"

# Split X, y
X = df.drop(columns=[target_col]).values
y = df[target_col].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CNN expects [batch, channel, length]
X_scaled = np.expand_dims(X_scaled, axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

import torch
from torch.utils.data import Dataset, DataLoader

class BatteryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = BatteryDataset(X_train, y_train)
test_data = BatteryDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

import torch.nn as nn

class BatteryCNN(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear((input_len // 2) * 32, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

model = BatteryCNN(input_len=X_scaled.shape[2])

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss = {running_loss/len(train_loader):.4f}")


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

model.eval()
preds, actuals = [], []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs).squeeze()
        preds.extend(outputs.tolist())
        actuals.extend(targets.tolist())

preds = np.array(preds)
actuals = np.array(actuals)

# Metrics
rmse = np.sqrt(mean_squared_error(actuals, preds))
mae = mean_absolute_error(actuals, preds)
r2 = r2_score(actuals, preds)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# Plot
plt.figure(figsize=(10,4))
plt.plot(actuals[:100], label="True RUL")
plt.plot(preds[:100], label="Predicted RUL", linestyle='--')
plt.xlabel("Sample Index")
plt.ylabel("RUL")
plt.title("RUL Prediction (CNN model)")
plt.legend()
plt.grid(True)
plt.show()

import torch

# Make sure model is in eval mode
model.eval()

# Create a dummy input with the same shape as your training data
dummy_input = torch.randn(1, 1, X_scaled.shape[2])

# Export the model
torch.onnx.export(
    model,                        # model being run
    dummy_input,                  # model input (or a tuple for multiple inputs)
    "battery_cnn.onnx",           # where to save the model
    input_names=["input"],        # the model's input names
    output_names=["output"],      # the model's output names
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable batch size
    opset_version=17              # ONNX version
)

print("ONNX model exported successfully!")
