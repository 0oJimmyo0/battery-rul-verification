import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
df = pd.read_csv("/Users/wuwenrong/Desktop/Battery_dataset.csv")
df = df.drop(columns=["battery_id"])
target_col = "RUL"

X = df.drop(columns=[target_col]).values
y = df[target_col].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.expand_dims(X_scaled, axis=1)  # [samples, channel, length]

seq_len = X_scaled.shape[2]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -----------------------------
# 2. Define CNN compatible with αβ-CROWN
# -----------------------------
class BatteryCNN_Verifiable(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            dummy = self.relu(self.conv1(dummy))
            dummy = self.relu(self.conv2(dummy))
            self.flattened_size = dummy.numel() // dummy.shape[0]

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


model = BatteryCNN_Verifiable(seq_len)
model.train()  # required for αβ-CROWN parsing

# -----------------------------
# 3. Wrap in αβ-CROWN
# -----------------------------
dummy_input = torch.randn(1, 1, seq_len)
bounded_model = BoundedModule(model, dummy_input, device='cpu')

# -----------------------------
# 4. Define perturbation
# -----------------------------
eps = 0.01
perturbation = PerturbationLpNorm(norm=float('inf'),
                                  x_L=dummy_input - eps,
                                  x_U=dummy_input + eps)
bounded_input = BoundedTensor(dummy_input, perturbation)

# -----------------------------
# 5. Compute bounds
# -----------------------------
lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method="IBP+backward")
print("Verified lower bound:", lb)
print("Verified upper bound:", ub)

# -----------------------------
# 6. Monotonicity check (cycle feature at index 0)
# -----------------------------
delta = 0.01
input_increased = dummy_input.clone()
input_increased[0, 0, 0] += delta  # increase cycle feature

output_orig = model(dummy_input)
output_increased = model(input_increased)

print("Original output:", output_orig.item())
print("Output after increasing cycle:", output_increased.item())
print("Monotonicity satisfied?", output_increased.item() >= output_orig.item())
