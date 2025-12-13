import numpy as np
import onnxruntime as ort

# -----------------------------
# Load ONNX model
# -----------------------------
onnx_path = r"C:\Users\qq293\PycharmProjects\PythonProject\rul_mlp.onnx"
print(f" Loading ONNX: {onnx_path}")

session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input name = {input_name}")
print(f"Output name = {output_name}")

# -----------------------------
# Predict one sample at a time
# -----------------------------
def predict_one(x):
    """ x must be shape (9,) """
    x = x.reshape(1, 9).astype(np.float32)
    y = session.run([output_name], {input_name: x})[0]
    return y[0][0]

# -----------------------------
# Basic test samples (random)
# -----------------------------
print("\n Generating sample points for verification...")
test_samples = np.random.rand(30, 9)  # 30 test points
print(f" Test samples shape: {test_samples.shape}")

# -----------------------------
# Verification 1: output >= 0
# -----------------------------
def verify_non_negative(samples):
    print("\n Verifying: RUL ≥ 0 ...")
    for i, s in enumerate(samples):
        y = predict_one(s)
        print(f"Sample {i:02d} → Output = {y:.4f}")
        if y < 0:
            print("FAIL: Negative output found!")
            return False
    print("PASS: All outputs are non-negative!\n")
    return True

# -----------------------------
# Verification 2: Output stability under ε noise
# -----------------------------
def verify_robustness(samples, eps=0.01):
    print("\n Verifying robustness (±ε noise)...")
    for i, s in enumerate(samples):
        y0 = predict_one(s)
        noise = (np.random.rand(9) * 2 - 1) * eps
        s2 = np.clip(s + noise, 0, 1)
        y1 = predict_one(s2)
        diff = abs(y1 - y0)
        print(f"Sample {i:02d} → Δ={diff:.6f}")
        if diff > 1.0:
            print("FAIL: Too sensitive to noise!")
            return False
    print("PASS: Model is stable under small perturbations!\n")
    return True

# -----------------------------
# Verification 3: Monotonicity in cycle dimension
# cycle = feature 0
# -----------------------------
def verify_monotonicity():
    print("\n Verifying monotonicity: cycle₁ < cycle₂ ⇒ RUL₁ ≥ RUL₂")
    base = np.random.rand(1, 9)[0]
    base[0] = 0.2
    s1 = base.copy()
    s2 = base.copy()
    s2[0] = 0.9  # Bigger cycle

    y1 = predict_one(s1)
    y2 = predict_one(s2)

    print(f"Cycle small → RUL={y1:.4f}")
    print(f"Cycle large → RUL={y2:.4f}")

    if y1 >= y2:
        print("PASS: Monotonicity holds!\n")
        return True
    else:
        print("FAIL: RUL increased with cycle!\n")
        return False

# -----------------------------
# Verification 4: Output upper bound
# -----------------------------
def verify_upper_bound(samples, max_rul=3.73):
    print(f"\n Verifying: RUL ≤ {max_rul} ...")
    for i, s in enumerate(samples):
        y = predict_one(s)
        if y > max_rul:
            print(f"FAIL: Output too large! y={y:.4f}")
            return False
    print("PASS: Output always within upper bound!\n")
    return True

# -----------------------------
# Run all verifications
# -----------------------------
verify_non_negative(test_samples)
verify_robustness(test_samples)
verify_monotonicity()
verify_upper_bound(test_samples)
