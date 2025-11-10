import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt

# ====== CONFIG ======
PATH = ("/home/rafa/PycharmProjects/COVID_BA/DATA/PA_Longitudinal.csv")  # <-- fixed extension
ID_CANDIDATES = ["ID_t1"]  # will pick the first that exists
GROUP_COL = "Group"                         # optional; used to keep only COV
T1_COL_CANDIDATES = ["BrainPAD_c_t1"]
T2_COL_CANDIDATES = ["BrainPAD_c_t2"]
ALPHA = 0.05
N_BOOT = 5000
SEED = 42
SAVE_BA_PLOT = False
BA_PLOT_PATH = "bland_altman_cov.png"

rng = np.random.default_rng(SEED)

# ====== LOAD ======
df = pd.read_csv(PATH)

# Pick ID column robustly
def pick_col(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No {name} column found. Looked for: {candidates}. Available: {df.columns.tolist()}")

ID_COL = pick_col(df, ID_CANDIDATES, "ID")
T1 = pick_col(df, T1_COL_CANDIDATES, "t1")
T2 = pick_col(df, T2_COL_CANDIDATES, "t2")

# Keep only COV patients if Group exists
if GROUP_COL in df.columns:
    df = df[df[GROUP_COL].astype(str).str.contains("cov", case=False, na=False)].copy()

# Keep complete pairs (one per ID)
wide = df[[ID_COL, T1, T2]].dropna().copy()
wide = wide.drop_duplicates(subset=ID_COL, keep="first")
x = wide[T1].astype(float).to_numpy()
y = wide[T2].astype(float).to_numpy()
n = len(wide)
if n < 2:
    raise ValueError("Need at least 2 complete pairs.")

# ====== ICC(3,1): two-way mixed, consistency, single measurement ======
def icc_3_1_from_pairs(x, y):
    """Return ICC(3,1) and ANOVA components (MSR, MSC, MSE, dfs) from paired vectors."""
    data = np.column_stack([x, y])  # shape (n, k) with k=2
    n, k = data.shape
    mean_targets = data.mean(axis=1, keepdims=True)
    mean_raters  = data.mean(axis=0, keepdims=True)
    grand_mean   = data.mean()

    # Sum of squares
    SSR = k * np.sum((mean_targets.squeeze() - grand_mean) ** 2)               # rows/subjects
    SSC = n * np.sum((mean_raters.squeeze()  - grand_mean) ** 2)               # columns/raters
    SSE = np.sum((data - mean_targets - mean_raters + grand_mean) ** 2)        # residual

    dfR = n - 1
    dfC = k - 1
    dfE = (n - 1) * (k - 1)

    MSR = SSR / dfR if dfR > 0 else np.nan
    MSC = SSC / dfC if dfC > 0 else 0.0
    MSE = SSE / dfE if dfE > 0 else np.nan

    icc = (MSR - MSE) / (MSR + (k - 1) * MSE)  # ICC(3,1) consistency
    return float(icc), float(MSR), float(MSC), float(MSE), int(dfR), int(dfC), int(dfE)

icc, MSR, MSC, MSE, dfR, dfC, dfE = icc_3_1_from_pairs(x, y)
SEM = float(np.sqrt(MSE))                   # standard error of measurement
MDC95 = float(1.96 * np.sqrt(2) * SEM)      # minimal detectable change at 95%

# ----- Bootstrap 95% CI for ICC -----
icc_boot = []
idx = np.arange(n)
for _ in range(N_BOOT):
    b = rng.choice(idx, size=n, replace=True)
    icc_b, *_ = icc_3_1_from_pairs(x[b], y[b])
    if np.isfinite(icc_b):
        icc_boot.append(icc_b)
icc_boot = np.array(icc_boot, dtype=float)
icc_ci_low, icc_ci_high = np.quantile(icc_boot, [0.025, 0.975]) if icc_boot.size else (np.nan, np.nan)

# ====== Spearman rank correlation (t1 vs t2) + bootstrap CI ======
rho, rho_p = st.spearmanr(x, y, nan_policy="omit")

rho_boot = []
for _ in range(N_BOOT):
    b = rng.choice(idx, size=n, replace=True)
    rb, _ = st.spearmanr(x[b], y[b])
    if np.isfinite(rb):
        rho_boot.append(rb)
rho_boot = np.array(rho_boot, dtype=float)
rho_ci_low, rho_ci_high = np.quantile(rho_boot, [0.025, 0.975]) if rho_boot.size else (np.nan, np.nan)

# ====== Bland–Altman stats ======
diff = y - x
avg  = (x + y) / 2
bias = float(diff.mean())
sd_d = float(diff.std(ddof=1))
loa_low  = bias - 1.96 * sd_d
loa_high = bias + 1.96 * sd_d

# ====== Print summary ======
print(f"n pairs = {n}")
print(f"ICC(3,1) = {icc:.3f}  (95% CI: {icc_ci_low:.3f}, {icc_ci_high:.3f})")
print(f"SEM = {SEM:.3f}  |  MDC95 = {MDC95:.3f}")
print(f"Spearman rho = {rho:.3f} (95% CI: {rho_ci_low:.3f}, {rho_ci_high:.3f}), p = {rho_p:.3g}")
print(f"Bland–Altman bias = {bias:.3f}, SD(diff) = {sd_d:.3f}")
print(f"95% LoA = [{loa_low:.3f}, {loa_high:.3f}]")

# ====== (Optional) Bland–Altman plot ======
plt.figure(figsize=(6,4))
plt.scatter(avg, diff, alpha=0.65, edgecolor="none")
plt.axhline(bias, linestyle="--")
plt.axhline(loa_low, linestyle=":")
plt.axhline(loa_high, linestyle=":")
plt.xlabel("Mean of t1 and t2 (age-corrected BAG)")
plt.ylabel("t2 − t1")
plt.title("Bland–Altman (COV patients)")
plt.tight_layout()
plt.show()  # uncomment to display
