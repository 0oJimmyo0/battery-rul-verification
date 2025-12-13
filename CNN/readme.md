
# CNN Training and Monotonicity Verification with α-β-CROWN

This project implements a complete pipeline for training a CNN-based
Remaining Useful Life (RUL) prediction model and verifying its monotonicity
property using the α-β-CROWN neural network verifier.

The workflow consists of two main phases:
1) Training a CNN model in Google Colab  
2) Formally verifying monotonicity using α-β-CROWN on a local machine

---

## Step 1: Train the CNN Model in Google Colab

1. Open **Google Colab**.
2. Upload and run the training script:
   ```text
   CNN_train.py
````

3. The script will:

   * Load the battery dataset
   * Train a CNN regression model for RUL prediction
   * Export the trained model to an **ONNX** file

Ensure the script completes successfully and produces an ONNX model


---

## Step 2: Download the ONNX Model

After training completes in Colab:

1. Locate the generated `.onnx` file in the Colab file browser.
2. Download it to your local machine:

   ```text
   cnn_battery_model.onnx
   ```

This ONNX file will be used as input to the verification step.

---

## Step 3: Download α-β-CROWN

Clone the official α-β-CROWN repository:

```bash
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN
cd alpha-beta-CROWN
```

This repository contains the verification engine used to certify neural network
properties such as monotonicity.

---

## Step 4: Create a Virtual Environment

It is strongly recommended to use a clean virtual environment for verification.

### Option A: Using `venv`

```bash
python3 -m venv abc_env
source abc_env/bin/activate
```

### Option B: Using `conda`

```bash
conda create -n abc_env python=3.9
conda activate abc_env
```

---

## Step 5: Install Required Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If additional dependencies are needed (e.g., for ONNX or PyTorch), install them
manually:

```bash
pip install torch onnx onnxruntime numpy
```

---

## Step 6: Run α-β-CROWN Verification

1. Copy the downloaded ONNX model into the α-β-CROWN directory
   (or update the model path inside the script).
2. Run the verification script:

```bash
python abc.py
```

This script will:

* Load the CNN model from the ONNX file
* Apply α-β-CROWN bound propagation
* Verify the monotonicity property with respect to the **cycle** feature

A successful run will report verified output bounds and indicate whether
the monotonicity constraint is satisfied.

---

## Expected Output

* Verified lower and upper bounds on the CNN output
* Comparison between original and perturbed inputs
* A boolean result confirming whether monotonicity is satisfied

---

## Notes

* Training and verification are intentionally separated to simplify environment
  management.
* Verification guarantees are local to the specified input bounds.
* Larger models may require additional parameter tuning for scalability.

---

## References

* α-β-CROWN: [https://github.com/Verified-Intelligence/alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
* AutoLiRPA verification framework

```

---

If you want, I can also:
- Add a **diagram of the pipeline**
- Add a **troubleshooting section** for common Colab/ONNX issues
- Rewrite this as a **course-submission README**
- Generate a matching **`requirements.txt`**

Just tell me.
```
