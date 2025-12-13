import numpy as np
import matplotlib.pyplot as plt

# --- Synthetic cycle numbers (115 to 250) ---
cycles = np.arange(115, 251)

# --- Create synthetic true RUL (oscillating like your plot) ---
np.random.seed(0)
true_rul = 120 + 60*np.sin(cycles/6) + 40*np.sin(cycles/3) + np.random.normal(0, 12, len(cycles))

# --- Create synthetic predicted RUL (similar shape but smoother) ---
pred_rul = 120 + 55*np.sin(cycles/6 + 0.2) + 35*np.sin(cycles/3 + 0.1) + np.random.normal(0, 10, len(cycles))

# Clip values to realistic range
true_rul = np.clip(true_rul, 0, 250)
pred_rul = np.clip(pred_rul, 0, 250)

# --- Plot with inverse y-axis ---
plt.figure(figsize=(12,5))
plt.plot(cycles, true_rul, label="True RUL")
plt.plot(cycles, pred_rul, label="Predicted RUL", linestyle='--')

plt.gca().invert_yaxis()   # <<< Flip RUL axis (high→low)

plt.xlabel("Cycle Number")
plt.ylabel("Remaining Useful Life (RUL)")
plt.title("CNN RUL Prediction vs Cycle Number (Inverted Y-axis)")
plt.legend()
plt.grid(True)
plt.show()