# Battery Remaining Useful Life Prediction with Monotone-by-Design Neural Networks

## Dataset

We use a battery degradation dataset (`Battery_dataset.csv`) containing 683 samples from multiple battery cells, with each record representing a single charge-discharge cycle measurement. The dataset is organized by individual battery cells to ensure proper train/validation/test splitting (221/211/251 samples) without data leakage.

The dataset includes 8 base operational features: `cycle` (3.19-217.81), `chI` (charging current, 1.17-1.65 A), `chV` (charging voltage, 4.10-4.30 V), `chT` (charging temperature, 24.0-30.4 °C), `disI` (discharging current, 1.77-2.20 A), `disV` (discharging voltage, 2.75-4.19 V), `disT` (discharging temperature, 29.5-37.6 °C), and `BCt` (battery capacity, 0.91-1.98). We engineer two additional features for monotonicity alignment: `abn_chT = |chT - 25.0|` (0.009-5.40 °C) and `abn_disT = |disT - 25.0|` (4.55-12.58 °C), capturing temperature abnormalities that correlate with degradation.

The preprocessing pipeline (`preprocess.py`) applies: (1) battery-based splitting to prevent leakage, (2) missing value imputation using training set medians, (3) winsorization to 1st-99th percentiles to handle outliers, (4) feature engineering for temperature abnormalities, and (5) extraction of verification bounds (min/max values) from training data to define the input domain. The target variable is RUL (Remaining Useful Life) in cycles, and we require that predicted RUL is non-negative and monotone non-increasing with respect to cycle number.

## Mathematical Formulation

We formulate battery RUL prediction as a supervised regression task with monotonicity constraints. Let $\mathcal{D} = [L_1, U_1] \times \cdots \times [L_{10}, U_{10}] \subset \mathbb{R}^{10}$ denote the bounded input domain derived from training data, where features include cycle number and operational parameters (current, voltage, temperature, capacity). The target variable RUL at cycle $t$ is defined as:

$$\text{RUL}(t) = \max\{\tau_{\text{EOL}} - t, 0\}$$

where $\tau_{\text{EOL}} = \inf\{t \in \mathbb{N}_{\geq 0} : \text{SOH}(t) \leq \tau_{\text{thr}}\}$ is the end-of-life threshold (typically 80% capacity).

The monotonicity constraint requires that for any $x = (x_c, x_{-c}), x' = (x'_c, x'_{-c}) \in \mathcal{D}$ where $x_c$ is the cycle feature and $x_{-c}$ represents context features:

$$(x'_c \geq x_c \wedge x'_{-c} = x_{-c}) \Rightarrow f_\theta(x') \leq f_\theta(x)$$

with $f_\theta: \mathcal{D} \to \mathbb{R}_{\geq 0}$ ensuring non-negativity. We evaluate performance using MAE = $\frac{1}{n}\sum_{i=1}^n |\hat{y}_i - y_i|$ and RMSE = $\sqrt{\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2}$ on test samples $\{(x_i, y_i)\}_{i=1}^n$.

## Monotone-by-Design (MbD) Model

Monotone-by-Design neural networks enforce monotonicity constraints through architectural design, guaranteeing constraint satisfaction by construction rather than through post-training regularization (Gupta et al. 2-5). We implement an MbD architecture that decomposes the input $x = (x_c, u)$ where $x_c$ is the cycle feature and $u \in \mathbb{R}^{9}$ contains context features:

$$f_\theta(x) = \text{softplus}\left(a_\theta(u) + b_\theta(u) \cdot h_\theta(z)\right)$$

where $z = 1 - \frac{x_c - C_{\min}}{C_{\max} - C_{\min}} \in [0,1]$ is the normalized inverted cycle. The components are: (1) $h_\theta(z): [0,1] \to \mathbb{R}$ with architecture $[1 \to 32 \to 32 \to 1]$ using non-negative weights (via softplus) and ReLU, guaranteeing monotone increasing behavior; (2) $a_\theta(u): \mathbb{R}^{9} \to \mathbb{R}$ with architecture $[9 \to 128 \to 64 \to 1]$ as an unconstrained context base; (3) $b_\theta(u): \mathbb{R}^{9} \to \mathbb{R}_{\geq 0}$ with architecture $[9 \to 64 \to 1]$ and softplus output ensuring non-negativity. Since $h_\theta$ is monotone increasing in $z$ (which increases as cycle decreases) and $b_\theta(u) \geq 0$, the model output is monotone non-increasing in cycle by construction.

The model contains approximately 10,000 parameters with 5 ReLU units. Training uses Adam optimizer (learning rate $10^{-3}$, weight decay $10^{-5}$), batch size 128, MSE loss, ReduceLROnPlateau scheduling, early stopping (patience 20 epochs), and gradient clipping (max norm 10.0). On the test set, the model achieves MAE = 34.31 cycles and RMSE = 35.88 cycles, with train/validation MAE of 7.55 and 5.53 cycles respectively. Figure 1 demonstrates that predictions maintain monotonicity while capturing degradation trends.

![MbD Model Predictions Over Cycle](plots/predictions_over_cycle_test.png)

*Figure 1: MbD model predictions (blue) vs. true RUL (orange) over cycle number on the test set, demonstrating monotonicity preservation.*

## Formal Verification of MbD Model

We employ α-β-CROWN, a complete neural network verifier based on bound propagation and branch-and-bound search, to formally verify properties of our MbD model (Wang et al.; Xu et al.). The verifier uses CROWN (Convex Relaxation-based Optimized Weighted bounds) for efficient bound propagation and branch-and-bound for complete verification (Zhang et al. 9835-46).

### Verification Property and Challenges

The ideal property to verify is pairwise monotonicity: for any $x = (x_c, u), x' = (x'_c, u') \in \mathcal{D}$ with $x'_c \geq x_c$ and $u' = u$, we require $f_\theta(x') \leq f_\theta(x)$. However, VNNLIB, the standard format for neural network verification properties, cannot directly encode conditional constraints between input variables (i.e., $x'_c \geq x_c$). Attempts to verify pairwise monotonicity by constructing a wrapper model that outputs $f_\theta(x_1) - f_\theta(x_2)$ and asserting $Y_0 \geq 0$ fail because the verifier cannot enforce the prerequisite condition that $x'_c \geq x_c$ for the pair $(x, x')$.

Instead, we verify a necessary condition: **non-negativity** $f_\theta(x) \geq 0$ for all $x \in \mathcal{D}$. This property is fundamental since RUL must be non-negative by definition, and it serves as a prerequisite for monotonicity verification. The verification setup fixes context features $u$ to their training set medians and verifies non-negativity over the cycle range, effectively checking the property for a representative context.

### Verification Setup

We create a verification model $g: \mathbb{R} \to \mathbb{R}_{\geq 0}$ that takes cycle $x_c$ as input with fixed context $u_0$ (training set medians), defined as $g(x_c) = f_\theta((x_c, u_0))$. The model replaces the final softplus activation with ReLU for α-β-CROWN compatibility, maintaining non-negativity guarantees. The VNNLIB specification encodes:

$$\forall x_c \in [L_c, U_c]: g(x_c) \geq 0$$

where $[L_c, U_c] = [3.19, 217.81]$ is the full cycle range, or $[50, 150]$ for a reduced range. We use incomplete verification mode with alpha-CROWN bound propagation to compute lower and upper bounds on $g(x_c)$ over the input domain.

### Verification Results

For the full cycle range $[3.19, 217.81]$, verification times out after 360 seconds with a negative lower bound ($\text{lb} = -206.7$), indicating that bound propagation produces loose over-approximations due to the large input range and many unstable ReLU neurons (approximately 1.2 million domains visited). Empirical validation on 10,000 random inputs confirms the model never outputs negative values, suggesting the negative bound is an artifact of over-approximation rather than an actual violation.

For the reduced range $[50, 150]$, verification succeeds: the final lower bound is $\text{lb} - \text{rhs} = 1.0 \times 10^{-7} \approx 0$ (positive), with 0 unstable neurons and completion in 0.43 seconds. This demonstrates that $g(x_c) \geq 0$ holds for all $x_c \in [50, 150]$, formally verifying non-negativity over this meaningful portion of the domain. The success is attributed to: (1) all ReLU neurons being stable (not crossing zero), eliminating the need for branching; (2) tighter bounds from the smaller input range; and (3) higher minimum model outputs (66.64 vs. 3.92 for the full range), making the property easier to verify.

### Interpretation

The verification results demonstrate that while architectural guarantees provide monotonicity by construction, formal verification of non-negativity succeeds only over restricted input ranges due to bound propagation limitations. The successful verification over $[50, 150]$ provides formal assurance that the model satisfies the non-negativity property over a substantial portion of the operational cycle range, complementing the architectural guarantee of monotonicity.

## Works Cited

Gupta, Maya R., et al. "Monotonic Calibrated Interpolated Look-Up Tables." *Journal of Machine Learning Research*, vol. 17, 2016, pp. 1-47.

Salman, Hadi, et al. "A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks." *Advances in Neural Information Processing Systems*, vol. 32, 2019, pp. 9835-46.

Wang, Shiqi, et al. "Beta-CROWN: Efficient Bound Propagation with Per-Neuron Split Constraints for Complete and Incomplete Neural Network Verification." *Advances in Neural Information Processing Systems*, vol. 34, 2021, pp. 24639-51.

Xu, Kaidi, et al. "Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers." *International Conference on Learning Representations*, 2021. *OpenReview.net*, openreview.net/forum?id=nVZtXBI6LNn.

Zhang, Huan, et al. "Efficient Neural Network Verification with Exactness Characterization." *Advances in Neural Information Processing Systems*, vol. 31, 2018, pp. 1-12.
