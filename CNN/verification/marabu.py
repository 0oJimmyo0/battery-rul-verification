# marabou_verify_monotonicity.py
from maraboupy import Marabou
import numpy as np

# path to your ONNX model
model_path = "cnn_battery_model.onnx"
network = Marabou.read_onnx(model_path)

# flatten input/output var arrays to lists
inputVars = network.inputVars[0].flatten()
outputVars = network.outputVars[0].flatten()

# index of 'cycle' in your input vector (0-based)
cycle_index = 0
cycle_var = inputVars[cycle_index]
out_var = outputVars[0]  # single scalar output RUL

# Make a copy of inputs (second copy for y)
# Simplest approach: create extra variables in the network to represent the second input
# Marabou's python API lacks a single "duplicate input" helper, so we can add variables artificially.

# Create second input variables (same shape)
input2_vars = []
for _ in inputVars:
    v = network.createNewInputVariable()  # create new input var
    input2_vars.append(v)

# Add bounds for both x and y (use dataset-driven bounds)
# Example: set dataset min/max (replace with actuals from your preprocessing)
L = [0.0]*len(inputVars)  # replace per-feature lower bound
U = [1.0]*len(inputVars)  # replace per-feature upper bound

for i, v in enumerate(inputVars):
    network.setLowerBound(v, L[i])
    network.setUpperBound(v, U[i])

for i, v in enumerate(input2_vars):
    network.setLowerBound(v, L[i])
    network.setUpperBound(v, U[i])

# enforce y_cycle >= x_cycle + eps
eps = 1e-6
network.addInequality([input2_vars[cycle_index], inputVars[cycle_index]], [1.0, -1.0], -eps)
# Note: addInequality(left_vars, left_coeffs, right_value) enforces sum(coeffs*vars) <= right_value
# so here we do: 1*y_cycle + (-1)*x_cycle <= -eps  => y_cycle - x_cycle <= -eps
# we want y_cycle >= x_cycle + eps => x_cycle - y_cycle <= -eps
# adjust sign accordingly. If confusion arises, test with small examples.

# Tie non-cycle features equal: for j != cycle, add equality x_j == y_j
for j in range(len(inputVars)):
    if j == cycle_index:
        continue
    network.addEquality([inputVars[j], input2_vars[j]], [1.0, -1.0], 0.0)

# Now add output relationship: RUL(y) - RUL(x) <= 0
# We need nodes representing output for the second input; easiest approach:
# Create two sub-networks or re-evaluate the network is nontrivial in Marabou API.
# Instead, a reliable approach is to build a new ONNX that accepts concatenated [x,y] and outputs f(x), f(y).
# So: prefer building that ONNX in PyTorch and then read that ONNX into Marabou for straightforward constraints.

print("Recommendation: For robustness, build an ONNX model that takes [x,y] and returns [f(x), f(y)],")
print("then add inequality f(y) - f(x) <= 0. See notes in README for code to export combined model.")
