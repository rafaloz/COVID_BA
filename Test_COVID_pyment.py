from utils_Train import *
import pickle
import ast

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from utils_Harmonization import *
import configparser

import pingouin as pg

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro, levene

from scipy.stats import gaussian_kde

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Opcional (si tienes SciPy para tests):
try:
    from scipy import stats
    SCIPY = True
except Exception:
    SCIPY = False

# Function to update BrainPAD
def update_prededad(results_df, predictions_df,
                    id_col_results='ID', id_col_pred='id',
                    pred_col='prediction'):
    r = results_df.copy()
    p = predictions_df.copy()

    # Normaliza tipos/espacios por si acaso
    r[id_col_results] = r[id_col_results].astype(str).str.strip()
    p[id_col_pred]    = p[id_col_pred].astype(str).str.strip()

    if len(p[id_col_pred][0]) == 3 and p[id_col_pred][0][2] == '0':
        p[id_col_pred] = pd.to_numeric(p[id_col_pred], errors='coerce').dropna().astype('int64').tolist()

    # Si hay IDs duplicados en las predicciones, quédate con el último (o la media)
    p = p.drop_duplicates(subset=[id_col_pred], keep='last')

    # Crea el mapeo y sustituye
    pred_map = p.set_index(id_col_pred)[pred_col]
    r['pred_Edad'] = r[id_col_results].map(pred_map)

    # Recalcula BrainPAD si existe Edad
    if 'Edad' in r.columns:
        r['BrainPAD'] = r['pred_Edad'] - r['Edad']

    # (Opcional) chequeo de IDs sin match
    missing = r[r['pred_Edad'].isna()][id_col_results].unique()
    print(f'IDs sin predicción: {len(missing)}')

    return r

def raincloud_plot(
    data_list, labels, width=0.35, jitter=0.06, bandwidth='scott',
    box_width=0.35, kde_alpha=0.4, point_size=20, mirror=True,
    kde_on_top=True, ax=None, show_means=True, annotate_means=True, show_n_in_mean=True
):
    """
    data_list: list of 1D arrays (each group)
    labels   : category labels
    width    : max horizontal half-width of KDE
    jitter   : horizontal jitter for points
    bandwidth: 'scott'|'silverman' or float
    mirror   : if True, draw symmetric KDE on both sides (violin-like)
    kde_on_top: if True, draw KDE after points/box so it sits on top
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))

    positions = np.arange(1, len(data_list) + 1, dtype=float)

    colors = ["#A74A4A", "#9D9D4F"]

    for i, data in enumerate(data_list):
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        pos = positions[i]
        n = int(data.size)

        color = colors[i]

        means = []

        # --- KDE ---
        if data.size < 2 or np.std(data) == 0:
            y = np.linspace(np.mean(data) - 1, np.mean(data) + 1, 200)
            dens = np.zeros_like(y)
        else:
            kde = gaussian_kde(data, bw_method=bandwidth)
            y = np.linspace(np.min(data) - 1.5*np.std(data),
                            np.max(data) + 1.5*np.std(data), 400)
            dens = kde(y)
            if dens.max() > 0:
                dens = dens / dens.max() * width

        # --- draw order: box+points first if KDE on top, else reverse ---
        if not kde_on_top:
            # KDE first (behind)
            _draw_kde(ax, y, dens, pos, kde_alpha=kde_alpha, mirror=mirror)

        # Slim box (slightly offset)
        bp = ax.boxplot(
            data, vert=True, positions=[pos],
            widths=box_width*0.6, showfliers=False, patch_artist=False,
            boxprops=dict(color='k', alpha=1.0),
            medianprops=dict(color='k', linewidth=1.6),
            whiskerprops=dict(color='k'),
            capprops=dict(color='k')
        )
        # Lower zorder so KDE can overlay if requested
        for elem in bp['boxes'] + bp['medians'] + bp['whiskers'] + bp['caps']:
            elem.set_zorder(1)

        # Jittered points (to the right a bit)
        xj = pos + np.random.uniform(-jitter, jitter, size=data.size)
        ax.scatter(xj, data, s=point_size, alpha=0.75, linewidths=0, zorder=2, color=color)

        # --- NEW: mean marker (+ optional annotation) ---
        if show_means:
            mu = float(np.mean(data)) if data.size else np.nan
            means.append(mu)
            # diamond marker with white fill and colored edge so it pops over KDE
            ax.scatter([pos], [mu], s=110, marker='D', facecolor='white',
                       edgecolor=color, linewidths=2, zorder=5)
            if annotate_means and n > 0:
                label = f'{mu:.2f} (n={n})' if show_n_in_mean else f'{mu:.2f}'
                ax.text(pos + width*0.6, mu, label,
                        va='center', ha='left', fontsize=9, color=color, zorder=6)

        if kde_on_top:
            _draw_kde(ax, y, dens, pos, kde_alpha=kde_alpha, mirror=mirror,
                      zorder=4, color=color, outline_only=True)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.5, len(data_list) + 0.5)
    ax.set_ylabel('Brain age gap (years)')
    ax.set_title('Brain age gap by acquisition (raincloud)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def _draw_kde(ax, y, dens, pos, kde_alpha=0.2, mirror=True, zorder=3, color=None, outline_only=False, lw=2, edgecolor=None):
    if outline_only:
        c = edgecolor or color or 'k'
        # right ridge
        ax.plot(pos + dens, y, lw=lw, color=c, zorder=zorder)
        # left ridge if mirrored
        if mirror:
            ax.plot(pos - dens, y, lw=lw, color=c, zorder=zorder)
        return
    # Left half
    ax.fill_betweenx(y, pos - dens, pos, facecolor=color, alpha=kde_alpha, linewidth=0, zorder=zorder)
    if mirror:
        # Right half (mirror)
        ax.fill_betweenx(y, pos, pos + dens, facecolor=color, alpha=kde_alpha, linewidth=0, zorder=zorder)


def _metrics(df, y_true='Edad', y_pred='pred_Edad'):
    """Compute MAE, Pearson r, and R² with basic NaN safety."""
    d = df[[y_true, y_pred]].dropna()
    y = d[y_true].to_numpy()
    yhat = d[y_pred].to_numpy()
    mae = np.mean(np.abs(yhat - y)) if y.size else np.nan
    r = np.corrcoef(yhat, y)[0, 1] if y.size else np.nan
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, r, r2

def plot_pred_vs_age_two_groups(
    df_controls, df_patients,
    x_col='Edad', y_corr_col='pred_Edad', y_raw_col='pred_Edad',
    labels=('Controls', 'Patients'),
    markers=('o', 'o'),
    outfile_base=None
):
    # ---- Metrics on UNCORRECTED predictions (pred_Edad) ----
    c_mae, c_r, c_r2 = _metrics(df_controls, y_true=x_col, y_pred=y_raw_col)
    p_mae, p_r, p_r2 = _metrics(df_patients, y_true=x_col, y_pred=y_raw_col)

    # ---- Data for scatter (CORRECTED predictions) ----
    c_plot = df_controls[[x_col, y_corr_col]].dropna()
    p_plot = df_patients[[x_col, y_corr_col]].dropna()

    # Axis bounds
    all_vals = np.concatenate([c_plot.to_numpy().ravel(), p_plot.to_numpy().ravel()])
    lo = np.floor(np.nanmin(all_vals) / 5) * 5
    hi = np.ceil(np.nanmax(all_vals) / 5) * 5

    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- Scatter ----
    ax.scatter(
        c_plot[x_col], c_plot[y_corr_col],
        s=120, marker=markers[0], color="#A74A4A",
        alpha=0.75, linewidths=0.5, edgecolors='white',
        label=f"{labels[0]} — MAE={c_mae:.2f} y, r={c_r:.2f}, R$^2$={c_r2:.2f}"
    )
    ax.scatter(
        p_plot[x_col], p_plot[y_corr_col],
        s=120, marker=markers[1], color="#609560",
        alpha=0.75, linewidths=0.5, edgecolors='white',
        label=f"{labels[1]} — MAE={p_mae:.2f} y, r={p_r:.2f}, R$^2$={p_r2:.2f}"
    )

    # ---- Regression lines (on corrected predictions) ----
    xgrid = np.linspace(lo, hi, 200)

    if len(c_plot) >= 2:
        m_c, b_c = np.polyfit(c_plot[x_col].values, c_plot[y_corr_col].values, 1)
        ax.plot(xgrid, m_c * xgrid + b_c, color="#A74A4A", linewidth=1, alpha=0.95, zorder=3)

    if len(p_plot) >= 2:
        m_p, b_p = np.polyfit(p_plot[x_col].values, p_plot[y_corr_col].values, 1)
        ax.plot(xgrid, m_p * xgrid + b_p, color="#609560", linewidth=1, alpha=0.95, zorder=3)

    # Identity line
    ax.plot([lo, hi], [lo, hi], '--', linewidth=1, color='0.35', zorder=0)

    # Aesthetics
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Real age')
    ax.set_ylabel('Predicted age')
    ax.legend(frameon=False, loc='upper left', fontsize=9)

    plt.tight_layout()

    if outfile_base:
        plt.savefig(f"{outfile_base}.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{outfile_base}.pdf", bbox_inches='tight')
        # Uncomment if you also want SVG:
        # plt.savefig(f"{outfile_base}.svg", bbox_inches='tight')

    return fig, ax

# --- 1) Emparejar por sujeto (base_id = ccovN) ---
def add_base_id(df):
    out = df.copy()
    out['base_id'] = np.char.partition(out['ID'].to_numpy(dtype='U', na_value=''), '_')[:, 0]
    return out


def model_evaluation_clean(X_test, results):

    file_path = os.path.join(carpeta_modelos, 'ModeloLatestNoHarmo', 'SimpleMLP_nfeats_245_fold_0.pkl')
    with open(file_path, 'rb') as file:
        regresor = pickle.load(file)

    pred_test_median_all = regresor.predict(X_test)
    pred_test_median = pred_test_median_all

    # bias correction
    df_bias_correction = pd.read_csv(os.path.join(carpeta_modelos, 'ModeloLatestNoHarmo', 'DataFrame_bias_correction_1.csv'))

    model = LinearRegression()
    model.fit(df_bias_correction[['edades_train']], df_bias_correction['pred_train'])

    slope = model.coef_[0]
    intercept = model.intercept_

    results['pred_Edad'] = pred_test_median
    results['pred_Edad_c'] = (pred_test_median - intercept) / slope
    results['BrainPAD'] = results['pred_Edad'] - results['Edad']
    results['BrainPAD_c'] = results['pred_Edad_c'] - results['Edad']

    return results

config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

carpeta_datos = config_parser.get("DATOS", "carpeta_datos")
predictionsPyment = config_parser.get("DATOS", "predictions_pyment")
carpeta_modelos = config_parser.get("MODELOS", "carpeta_modelos")

controles_COVID_results = pd.read_csv(os.path.join(carpeta_datos, 'harmonized', 'Harmonized_Controls_results.csv'))
Harmonized_COVID_results = pd.read_csv(os.path.join(carpeta_datos, 'harmonized', 'Harmonized_COVID_results.csv'))
pac_COVID_II_results = pd.read_csv(os.path.join(carpeta_datos, 'harmonized', 'Harmonized_Longitudinal_results.csv'))

prediccionesPymentControls = pd.read_csv(os.path.join(predictionsPyment, 'pyment_predictions_Controls.csv'))
prediccionesPymentCOVID = pd.read_csv(os.path.join(predictionsPyment, 'pyment_predictions_COVID.csv'))
prediccionesPymentCOV_I = pd.read_csv(os.path.join(predictionsPyment, 'pyment_predictions_COVID_I.csv'))
prediccionesPymentCOV_II = pd.read_csv(os.path.join(predictionsPyment, 'pyment_predictions_COVID_II.csv'))

# IDs baseline que tienen segunda adquisición
ids_long = (pac_COVID_II_results["ID"].astype(str).str.extract(r"^(\d+)").squeeze().dropna().astype(int).tolist())

# Filtra pac_COVID_results in-place (o crea una copia si prefieres)
pac_COVID_I_results = (Harmonized_COVID_results[Harmonized_COVID_results['ID'].isin(ids_long)].reset_index(drop=True))

print(pac_COVID_I_results.shape)

# Apply to all datasets
controles_COVID_results = update_prededad(controles_COVID_results, prediccionesPymentControls)
pac_COVID_results = update_prededad(Harmonized_COVID_results, prediccionesPymentCOVID)
pac_COVID_I_results = update_prededad(pac_COVID_I_results, prediccionesPymentCOV_I)
pac_COVID_II_results = update_prededad(pac_COVID_II_results, prediccionesPymentCOV_II)

print('######## Resultado Controles COVID ##########')
res = summarize_metrics(controles_COVID_results, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

print('######## Resultado Pacientes COVID ##########')
res = summarize_metrics(pac_COVID_results, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

print('######## Resultado Pacientes COVID t1 ##########')
res = summarize_metrics(pac_COVID_I_results, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

print('######## Resultado Pacientes COVID t2 ##########')
res = summarize_metrics(pac_COVID_II_results, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^ ARE THE GROUPS COMPARABLE IN AGE AND SEX (YES THEY ARE) ^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
# Check Normality for Age in both groups using the Shapiro-Wilk test
print("Checking Normality of Age Distribution:")
for group, name in [(controles_COVID_results, "Controls"), (pac_COVID_results, "COVID")]:
    stat, p_value = stats.kstest(group['Edad'], 'norm', args=(group['Edad'].mean(), group['Edad'].std()))
    print(f"{name} Group - Kolmogorov-Smirnov Test: Stat={stat}, P-value={p_value}")
    if p_value < 0.05:
        print(f"The age distribution for the {name} group does not appear to be normal.\n")
    else:
        print(f"The age distribution for the {name} group appears to be normal.\n")

# Check Equality of Variances using Levene's test
stat, p_value = stats.levene(controles_COVID_results['Edad'], pac_COVID_results['Edad'])
print(f"Levene’s Test for Equality of Variances: Stat={stat}, P-value={p_value}")
if p_value < 0.05:
    print("Variances of the age distributions between the groups are significantly different.\n")
else:
    print("No significant difference in variances of the age distributions between the groups.\n")

# Perform ANOVA
f_statistic, p_value = stats.f_oneway(controles_COVID_results['Edad'], pac_COVID_results['Edad'])

print(f"F-statistic: {f_statistic}, p-value: {p_value}")

t_stat, p_val = stats.ttest_ind(controles_COVID_results['Edad'], pac_COVID_results['Edad'], equal_var=True)

print(f"t.test: {f_statistic}, p-value: {p_value}")

# Check for Sex Comparability (Categorical Data) using a Chi-Square test
# First, create a contingency table for the 'Sex' column
# Count the occurrences of each 'Sex' category within each group
healthy_sex_counts = controles_COVID_results['sexo(M=1;F=0)'].value_counts()
COVID_sex_counts = pac_COVID_results['sexo(M=1;F=0)'].value_counts()

# Create a DataFrame to represent the contingency table
contingency_table = pd.DataFrame({'Healthy': healthy_sex_counts, 'COVID': COVID_sex_counts,})

chi2_stat, p_value_sex, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Square Test for Sex:")
print(f"Chi-square statistic: {chi2_stat}, P-value: {p_value_sex}")
if p_value_sex < 0.05:
    print("Significant differences in sex distribution between the groups.")
else:
    print("No significant differences in sex distribution between the groups.")

# Clean and count
h = controles_COVID_results['sexo(M=1;F=0)'].dropna().astype(int).value_counts().reindex([1,0], fill_value=0)
p = pac_COVID_results['sexo(M=1;F=0)'].dropna().astype(int).value_counts().reindex([1,0], fill_value=0)

# Build 2×2 table: rows = sex (M, F), cols = groups
table = np.array([[h[1], p[1]],
                  [h[0], p[0]]], dtype=int)

# Fisher's exact test (two-sided)
oddsratio, p_fisher = stats.fisher_exact(table, alternative='two-sided')
print("\nFisher's Exact Test for Sex:")
print(f"2x2 table (M/F by Healthy/COVID):\n{table}")
print(f"Odds ratio: {oddsratio:.3f}, P-value: {p_fisher:.4g}")
if p_fisher < 0.05:
    print("Significant differences in sex distribution between the groups.")
else:
    print("No significant differences in sex distribution between the groups.")


print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^ ARE THE BrainPAD COMPARABLE (YES THEY ARE) ^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

# Check Normality for Age in both groups using the Shapiro-Wilk test
print("Checking Normality of Age Distribution:")
for group, name in [(controles_COVID_results, "Controls"), (pac_COVID_results, "COVID")]:
    stat, p_value = stats.kstest(group['BrainPAD'], 'norm', args=(group['BrainPAD'].mean(), group['BrainPAD'].std()))
    print(f"{name} Group - Kolmogorov-Smirnov Test: Stat={stat}, P-value={p_value}")
    if p_value < 0.05:
        print(f"The age distribution for the {name} group does not appear to be normal.\n")
    else:
        print(f"The age distribution for the {name} group appears to be normal.\n")

# Check Equality of Variances using Levene's test
stat, p_value = stats.levene(controles_COVID_results['BrainPAD'], pac_COVID_results['BrainPAD'])
print(f"Levene’s Test for Equality of Variances: Stat={stat}, P-value={p_value}")
if p_value < 0.05:
    print("Variances of the age distributions between the groups are significantly different.\n")
else:
    print("No significant difference in variances of the age distributions between the groups.\n")

print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ BRAIN-PAD ANCOVA ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

controles_COVID_results["euler_total"] = controles_COVID_results["euler_total"]/2
pac_COVID_results["euler_total"] = pac_COVID_results["euler_total"]/2

# pick a reference to compute the scaling — controls is a good choice
ref = controles_COVID_results["euler_total"].astype(float)

mu = ref.mean()
sd = ref.std(ddof=0)  # population sd (matches sklearn's StandardScaler)

controles_COVID_results["euler_total_z"] = (controles_COVID_results["euler_total"] - mu) / sd
pac_COVID_results["euler_total_z"] = (pac_COVID_results["euler_total"] - mu) / sd

controles_COVID_results['eTIV'] = controles_COVID_results['eTIV'] / 1000000
pac_COVID_results['eTIV'] = pac_COVID_results['eTIV'] / 1000000

fig, ax = plot_pred_vs_age_two_groups(
    controles_COVID_results,
    pac_COVID_results,
    labels=('Controls', 'COV'),
    outfile_base=None  # e.g. 'age_scatter_nature'
)
plt.show()

controles_COVID_results['Group'] = 'Controls'
pac_COVID_results['Group'] = 'COVID'

merged_df = pd.concat([controles_COVID_results, pac_COVID_results], axis=0)
df = merged_df[['BrainPAD', 'sexo(M=1;F=0)', 'Group', 'Edad', 'pred_Edad', 'eTIV', 'euler_total_z']]
df.columns = ['BrainPAD', 'sexo', 'Group', 'Age', 'BrainAge', 'eTIV', 'euler_total_z']

# df.to_csv('PA_Baseline.scv', index=False)

# === 1) Fit the same ANCOVA with statsmodels OLS to retrieve residuals ===
# BrainPAD ~ Group + covariates
# (C(Group) forces categorical; remove C() if Group is already 0/1 numeric)
model = smf.ols('BrainPAD ~ C(Group) + Age + eTIV + sexo + euler_total_z', data=df).fit()

# --- 1) Adjusted group difference (Patients–Controls), CI, p ---
coef_name = [c for c in model.params.index if c.startswith("C(Group)")][0]  #
adj_diff  = model.params[coef_name]
ci_low, ci_high = model.conf_int().loc[coef_name]
p_val     = model.pvalues[coef_name]
t_val     = model.tvalues[coef_name]
df_resid  = int(model.df_resid)

print(f"Adjusted Δ (Patients–Controls) = {adj_diff:.2f} y "
      f"(95% CI {ci_low:.2f} to {ci_high:.2f}); t({df_resid}) = {t_val:.2f}, p = {p_val:.3f}")

# --- 2) Covariate-adjusted Cohen's d (using residual SD from ANCOVA) ---
sigma = np.sqrt(model.mse_resid)      # common SD
d_adj = adj_diff / sigma
J = 1 - 3/(4*df_resid - 1)            # small-sample correction
g_adj = J * d_adj
print(f"Adjusted Cohen's d = {d_adj:.2f} (Hedges' g = {g_adj:.2f})")

# --- 3) Bootstrap 95% CI for adjusted d ---
def bootstrap_d_adj(data, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        samp = data.sample(n=len(data), replace=True, random_state=int(rng.integers(1e9)))
        samp["Group"] = pd.Categorical(samp["Group"], categories=["Controls", "Patients"])
        fit = smf.ols(
            "BrainPAD ~ C(Group) + Age + eTIV + sexo + euler_total_z",
            data=samp
        ).fit()
        coef_b = [c for c in fit.params.index if c.startswith("C(Group")][0]
        diff_b = fit.params[coef_b]
        sigma_b = np.sqrt(fit.mse_resid)
        vals[b] = diff_b / sigma_b
    return np.percentile(vals, [2.5, 50, 97.5]), vals

d_ci, _ = bootstrap_d_adj(df)
print(f"Bootstrap 95% CI for d: {d_ci[0]:.2f} to {d_ci[2]:.2f} (median {d_ci[1]:.2f})")

# Fitted and residuals
fitted = model.fittedvalues
resid   = model.resid
# Studentized residuals are nicer for diagnostics
stud_resid = model.get_influence().resid_studentized_internal

print(model.summary())  # optional

# === 2) Normality of residuals ===
# Shapiro–Wilk (works best for n < ~500; otherwise expect small p’s)
W, p_shapiro = shapiro(stud_resid if len(stud_resid) <= 500 else np.random.choice(stud_resid, 500, replace=False))
# Jarque–Bera (skew/kurtosis-based; fine for larger n)
jb_stat, jb_p, skew, kurt = jarque_bera(resid)

print(f"\nNormality:")
print(f"  Shapiro–Wilk: W={W:.3f}, p={p_shapiro:.3g}")
print(f"  Jarque–Bera : JB={jb_stat:.2f}, p={jb_p:.3g}, skew={skew:.3f}, kurtosis(excess)={kurt:.3f}")

# === 3) Homoscedasticity (equal variance) ===
# Breusch–Pagan and White tests
bp_LM, bp_p, bp_F, bp_Fp = het_breuschpagan(resid, model.model.exog)

print(f"\nHomoscedasticity:")
print(f"  Breusch–Pagan: LM={bp_LM:.2f}, p={bp_p:.3g} | F={bp_F:.2f}, p={bp_Fp:.3g}")

# === 4) Diagnostic plots ===
# QQ plot of studentized residuals
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

sm.ProbPlot(stud_resid).qqplot(line='s', ax=ax[0])
ax[0].set_title('QQ plot (studentized residuals)')

# Residuals vs fitted
ax[1].scatter(fitted, stud_resid, alpha=0.7, edgecolor='none')
ax[1].axhline(0, color='k', lw=1)
ax[1].set_xlabel('Fitted values')
ax[1].set_ylabel('Studentized residuals')
ax[1].set_title('Residuals vs fitted')

plt.tight_layout()
plt.show()

bp_ctrl = controles_COVID_results['BrainPAD'].to_numpy(dtype=float)
bp_ctrl = bp_ctrl[np.isfinite(bp_ctrl)]

bp_pat = pac_COVID_results['BrainPAD'].to_numpy(dtype=float)
bp_pat = bp_pat[np.isfinite(bp_pat)]

fig, ax = plt.subplots(figsize=(6.5, 5.2))
raincloud_plot([bp_ctrl, bp_pat], labels=['Controls', 'COV'],
               bandwidth='silverman', kde_alpha=0.45, kde_on_top=True, mirror=True, ax=ax)
plt.tight_layout()
plt.show()

plot_brain_age_vs_age(df)

######################3

# IDs baseline que tienen segunda adquisición
ids_long = (pac_COVID_II_results["ID"].astype(str).str.extract(r"^(\d+)").squeeze().dropna().astype(int).tolist())
ids_long = [str(x) for x in ids_long]

# Filtra pac_COVID_results in-place (o crea una copia si prefieres)
pac_COVID_results = (pac_COVID_results[pac_COVID_results['ID'].isin(ids_long)].reset_index(drop=True))

t1 = add_base_id(pac_COVID_results).copy()
t2 = add_base_id(pac_COVID_II_results).copy()

t1['base_id'] = t1['base_id'].astype('Int64').astype(str).str.zfill(3)

# Si hubiera duplicados por sujeto en un mismo timepoint, nos quedamos con uno (aquí el primero)
t1 = t1.drop_duplicates(subset='base_id', keep='first')
t2 = t2.drop_duplicates(subset='base_id', keep='first')

# Merge ancho (una fila por sujeto)
wide = t1.merge(
    t2,
    on='base_id',
    suffixes=('_t1', '_t2'),
    how='inner'
)

# Asegurar tipos numéricos (por si acaso)
num_cols = ['Edad', 'pred_Edad', 'BrainPAD', 'eTIV']
for c in num_cols:
    wide[f'{c}_t1'] = pd.to_numeric(wide[f'{c}_t1'], errors='coerce')
    wide[f'{c}_t2'] = pd.to_numeric(wide[f'{c}_t2'], errors='coerce')

# wide.to_csv('PA_Longidtudinal.scv', index=False)

# --- 2) Scatter Edad vs Brain Age (pred_Edad_c) con líneas por sujeto ---
plt.figure(figsize=(6,6))

# Líneas por sujeto (dirección del cambio)
for _, row in wide.iterrows():
    plt.plot([row['Edad_t1'], row['Edad_t2']],
             [row['pred_Edad_t1'], row['pred_Edad_t2']],
             linewidth=0.8)

# Puntos T1 y T2 (marcadores distintos)
plt.scatter(wide['Edad_t1'], wide['pred_Edad_t1'], s=90, label='T1 (baseline)', color='#A74A4A')
plt.scatter(wide['Edad_t2'], wide['pred_Edad_t2'], s=90, label='T2 (follow-up)', color='#9D9D4F')

# Línea identidad y ejes
xmin = np.nanmin([wide['Edad_t1'].min(), wide['Edad_t2'].min()])
xmax = np.nanmax([wide['Edad_t1'].max(), wide['Edad_t2'].max()])
ymin = np.nanmin([wide['pred_Edad_t1'].min(), wide['pred_Edad_t2'].min()])
ymax = np.nanmax([wide['pred_Edad_t1'].max(), wide['pred_Edad_t2'].max()])
lo = np.nanmin([xmin, ymin]); hi = np.nanmax([xmax, ymax])
plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1)

plt.xlabel('Edad cronológica')
plt.ylabel('Edad cerebral (pred_Edad)')
plt.title('Edad vs Edad cerebral con pares longitudinales')
plt.legend()
plt.tight_layout()
plt.show()

# --- 3) Test pareado BrainPAD (T1 vs T2) ---
from scipy.stats import ttest_rel, wilcoxon, t

# --- Pairwise-complete data ---
pairs = wide[['BrainPAD_t1', 'BrainPAD_t2']].dropna()
bp1 = pairs['BrainPAD_t1'].to_numpy()
bp2 = pairs['BrainPAD_t2'].to_numpy()
delta = bp2 - bp1
n = delta.size
df = n - 1

# --- Tests ---
tstat, p_t = ttest_rel(bp2, bp1)  # two-sided by default
wstat, p_w = wilcoxon(delta, zero_method='wilcox')  # excludes exact zeros

# --- Effect size (Cohen's d_z for paired data) ---
dz = delta.mean() / delta.std(ddof=1)
J = 1 - 3/(4*df - 1)              # small-sample correction (Hedges' g_z)
g_z = J * dz

# --- 95% CI for mean Δ (parametric; optional) ---
se = delta.std(ddof=1) / np.sqrt(n)
crit = t.ppf(0.975, df)
ci_mean = (delta.mean() - crit*se, delta.mean() + crit*se)

# --- Bootstrap 95% CI for d_z (resample pairs) ---
def boot_ci_dz_pairs(bp1, bp2, B=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = bp1.size
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)        # resample pairs
        d_b = (bp2[idx] - bp1[idx])
        vals[b] = d_b.mean() / d_b.std(ddof=1)
    lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
    return (lo, med, hi), vals

(dz_lo, dz_med, dz_hi), _ = boot_ci_dz_pairs(bp1, bp2)

print(f"N pairs: {n}")
print(f"BrainPAD T1:  mean={bp1.mean():.2f}, sd={bp1.std(ddof=1):.2f}")
print(f"BrainPAD T2:  mean={bp2.mean():.2f}, sd={bp2.std(ddof=1):.2f}")
print(f"Delta (T2-T1): mean={delta.mean():.2f}, sd={delta.std(ddof=1):.2f}, 95% CI [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")
print(f"Paired t-test: t={tstat:.2f}, p={p_t:.3g}")
print(f"Wilcoxon: W={wstat}, p={p_w:.3g}")
print(f"Cohen's d_z: {dz:.2f} (Hedges' g_z={g_z:.2f}), bootstrap 95% CI [{dz_lo:.2f}, {dz_hi:.2f}]")

# --- 4) Violin plot de BrainPAD T1 vs T2 ---
# Your data (pairwise complete)
bp1 = pairs['BrainPAD_t1'].to_numpy()
bp2 = pairs['BrainPAD_t2'].to_numpy()

bp1_arr = np.asarray(bp1)
bp2_arr = np.asarray(bp2)
mask = np.isfinite(bp1_arr) & np.isfinite(bp2_arr)
bp1_clean = bp1_arr[mask]
bp2_clean = bp2_arr[mask]

fig, ax = plt.subplots(figsize=(6.5, 5.2))
raincloud_plot([bp1_clean, bp2_clean], labels=['T1', 'T2'],
               bandwidth='silverman', kde_alpha=0.45, kde_on_top=True, mirror=True, ax=ax)
plt.tight_layout()
plt.show()