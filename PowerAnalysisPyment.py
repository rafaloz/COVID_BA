import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st

# =========================
# Utilidades generales
# =========================
def _twosided_power_from_nct(t_crit, df, ncp):
    # Potencia = P(|T| > t_crit) con T ~ nct(df, ncp)
    right = st.nct.sf(t_crit, df, ncp)
    left  = st.nct.cdf(-t_crit, df, ncp)
    return float(right + left)

def _bisect_int(lo, hi, func, tol=1e-4, maxit=60):
    # Binary search over integers for sample size.
    for _ in range(maxit):
        mid = (lo + hi) // 2
        if mid <= lo: break
        val = func(mid)
        if val < 0:  # below objective power
            lo = mid
        else:
            hi = mid
    return hi

# =========================
# 1) CASE–CONTROL (ANCOVA)
# =========================
def power_ancova_two_groups(delta, sd_y, n1, n2, *, alpha=0.05,
                            r2_cov=0.0, p_cov=0):
    """
    Power for detecting a mean difference between two groups in an ANCOVA,
    approximating the gain in precision as a variance reduction by (1 − R²_cov).
    - delta: mean difference you want to detect (e.g., 1.0 year in brain-age gap)
    - sd_y: standard deviation of the unadjusted outcome (e.g., raw BAG or similar)
    - n1, n2: sample sizes of Control and COV groups
    - r2_cov: R² of the model Y ~ covariates (excluding the group factor)
    - p_cov: number of covariates (for residual degrees of freedom: N − g − p)
    """
    N = n1 + n2
    g = 2
    df_err = N - g - p_cov
    if df_err <= 1:
        return np.nan

    sd_adj = sd_y * np.sqrt(1.0 - r2_cov)
    se = sd_adj * np.sqrt(1.0/n1 + 1.0/n2)
    ncp = delta / se
    t_crit = st.t.ppf(1 - alpha/2, df_err)
    return _twosided_power_from_nct(t_crit, df_err, ncp)

def sample_size_ancova_for_power(delta, sd_y, *, alpha=0.05, power_target=0.80,
                                 ratio=1.0, r2_cov=0.0, p_cov=0, n1_max=10000):
    """
    Returns the minimum (n1, n2, achieved_power) needed to reach power_target.
    ratio = n2/n1.
    """
    def gap(n1):
        n1 = int(n1)
        n2 = int(np.ceil(ratio * n1))
        pwr = power_ancova_two_groups(delta, sd_y, n1, n2,
                                      alpha=alpha, r2_cov=r2_cov, p_cov=p_cov)
        return pwr - power_target

    # Límite inferior razonable para df>0
    lo = max(4, p_cov + 4)
    hi = n1_max
    # Asegura que hay solución
    if gap(hi) < 0:
        return None  # ni con n grande alcanzas potencia (delta muy pequeño vs sd)

    n1_star = _bisect_int(lo, hi, gap)
    n2_star = int(np.ceil(ratio * n1_star))
    pwr = power_ancova_two_groups(delta, sd_y, n1_star, n2_star,
                                  alpha=alpha, r2_cov=r2_cov, p_cov=p_cov)
    return n1_star, n2_star, pwr

def mde_ancova(n1, n2, sd_y, *, alpha=0.05, power_target=0.80, r2_cov=0.0, p_cov=0):
    """
    Minimum detectable delta (mean difference) for the target power.
    """
    def gap(delta):
        pwr = power_ancova_two_groups(delta, sd_y, n1, n2,
                                      alpha=alpha, r2_cov=r2_cov, p_cov=p_cov)
        return pwr - power_target

    lo, hi = 1e-6, sd_y * 10
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return hi  # ≈ delta mínimo detectable

def estimate_sd_and_r2_from_df(df, y='BAG_raw', covars=('Edad','Sex','eTIV','Euler_z')):
    """
    Estimates sd_y and the R² of the covariates (excluding the group factor).
    """
    sd_y = df[y].std(ddof=1)
    formula = f"{y} ~ " + " + ".join(covars)
    fit = smf.ols(formula, data=df.dropna(subset=[y, *covars])).fit()
    return sd_y, float(fit.rsquared)

# =========================
# 2) LONGITUDINAL (PAREADO)
# =========================
def power_paired(mean_diff, sd_diff, n, *, alpha=0.05):
    """
    Power for a paired t-test (testing whether the mean change differs from zero).
    - mean_diff: expected mean change (e.g., 1.0 year)
    - sd_diff: standard deviation of within-subject differences
    """
    df = n - 1
    if df < 1: return np.nan
    se = sd_diff / np.sqrt(n)
    ncp = mean_diff / se
    t_crit = st.t.ppf(1 - alpha/2, df)
    return _twosided_power_from_nct(t_crit, df, ncp)

def sample_size_paired_for_power(mean_diff, sd_diff, *, alpha=0.05, power_target=0.80, n_max=10000):
    def gap(n):
        n = int(n)
        pwr = power_paired(mean_diff, sd_diff, n, alpha=alpha)
        return pwr - power_target

    lo, hi = 3, n_max
    if gap(hi) < 0:
        return None
    n_star = _bisect_int(lo, hi, gap)
    return n_star, power_paired(mean_diff, sd_diff, n_star, alpha=alpha)

def mde_paired(n, sd_diff, *, alpha=0.05, power_target=0.80):
    lo, hi = 1e-6, sd_diff * 10
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if power_paired(mid, sd_diff, n, alpha=alpha) < power_target:
            lo = mid
        else:
            hi = mid
    return hi

# =========================
# 3) MONTE CARLO (opcional)
# =========================
def mc_power_ancova_from_df(df, delta, n1, n2, *, alpha=0.05,
                            y='BAG_raw', group='Group',
                            covars=('Edad','Sex','eTIV','Euler_z'),
                            n_sims=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=[y, group, *covars]).copy()

    # Forzar referencia Control y asegurar dos niveles
    df[group] = pd.Categorical(df[group], categories=['Control','COV'])

    g0 = df[df[group]=='Control']
    g1 = df[df[group]=='COV']
    if len(g0) < 2 or len(g1) < 2:
        raise ValueError("There are not enough data per group to run the simulation.")

    formula = f"{y} ~ C({group}) + " + " + ".join(covars)
    hits = 0
    for _ in range(n_sims):
        s0 = g0.sample(n1, replace=True, random_state=rng.integers(1e9))
        s1 = g1.sample(n2, replace=True, random_state=rng.integers(1e9)).copy()
        s1[y] = s1[y] + delta  # efecto verdadero
        sim = pd.concat([s0, s1], ignore_index=True)
        fit = smf.ols(formula, data=sim).fit(cov_type='HC3')
        p = fit.pvalues["C(%s)[T.COV]" % group]
        if p < alpha:
            hits += 1
    return hits / n_sims


##############################################################################################

PA_Baseline = pd.read_csv('/home/rafa/PycharmProjects/COVID_github/DATA/PA_Baseline_pyment.csv')
PA_Longidtudinal = pd.read_csv('/home/rafa/PycharmProjects/COVID_github/DATA/PA_Longidtudinal_pyment.csv')

cols = ['BrainPAD','Group','Age','sex','eTIV','euler_number_z']
print("N total:", len(PA_Baseline))
print(PA_Baseline['Group'].value_counts(dropna=False))
print("NAs by column:\n", PA_Baseline[cols].isna().sum())

sd_y, r2 = estimate_sd_and_r2_from_df(PA_Baseline, y='BrainPAD', covars=('Age','sex','eTIV','euler_number_z'))
delta = 3.0
alpha = 0.05
p_cov = 4

# Power with current sample sizes (example: n1 = 80, n2 = 87)
n1, n2 = 48, 53
pwr = power_ancova_two_groups(delta, sd_y, n1, n2, alpha=alpha, r2_cov=r2, p_cov=p_cov)
print("power ANCOVA:", round(pwr, 3))

# Required sample size for 80% power (keeping the same ratio)
out = sample_size_ancova_for_power(delta, sd_y, alpha=alpha, power_target=0.80,
                                   ratio=n2/n1, r2_cov=r2, p_cov=p_cov)
print("n1*, n2*, power:", out)

# Minimum detectable effect with your current N
mde = mde_ancova(n1, n2, sd_y, alpha=alpha, power_target=0.80, r2_cov=r2, p_cov=p_cov)
print("MDE (years BAG):", round(mde, 2))

# (Optional) Empirical power via Monte Carlo using your dataframe
p_mc = mc_power_ancova_from_df(PA_Baseline, delta, n1, n2, alpha=alpha,
                               y='BrainPAD', group='Group',
                               covars=('Age','sex','eTIV','euler_number_z'),
                               n_sims=2000)
print("Power Monte Carlo:", round(p_mc, 3))


# df_long: one row per measurement; columns: ['ID', 'Time' in {'t1','t2'}, 'BAG_corr', ...]
wide = PA_Longidtudinal
dif = wide['BrainPAD_t2'] - wide['BrainPAD_t1']
n  = len(dif)
sd_diff = dif.std(ddof=1)

mean_diff = 3.0   # for example, you want to be able to detect a 1-year change
alpha = 0.05

# Power with your current n
pwr_long = power_paired(mean_diff, sd_diff, n, alpha=alpha)
print("Power t paired:", round(pwr_long, 3))

# Required sample size
n_star, pwr_star = sample_size_paired_for_power(mean_diff, sd_diff, alpha=alpha, power_target=0.80)
print("n* for 80%:", n_star, "power:", round(pwr_star, 3))

# MDE (with your n)
mde_long = mde_paired(n, sd_diff, alpha=alpha, power_target=0.80)
print("MDE longitudinal (years):", round(mde_long, 2))

print("n, sd_diff =", n, sd_diff)
print("Power at 1.0y:", power_paired(3.0, sd_diff, n))
d_mde = mde_paired(n, sd_diff, alpha=0.05, power_target=0.80)
print("MDE:", d_mde, "Power at MDE:", power_paired(d_mde, sd_diff, n))
for d in [0.5, 0.8, 1.0, 1.5]:
    print(d, "→", power_paired(d, sd_diff, n))


