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


def risk_summary_rr(df, bag_col='BrainPAD_c_t1', group_col='Grupo',
                    improve_label='Mejora', not_improve_label='No mejora',
                    bag_thr=0.0):
    d = df[[bag_col, group_col]].dropna().copy()
    d['BAGpos'] = (d[bag_col] > bag_thr).astype(int)
    d['NoMej']  = (d[group_col] == not_improve_label).astype(int)

    # Counts for 2x2
    # a = NoMej & BAGpos=1 ; b = Mejora & BAGpos=1 ; c = NoMej & BAGpos=0 ; d_ = Mejora & BAGpos=0
    a = int(((d['BAGpos']==1) & (d['NoMej']==1)).sum())
    b = int(((d['BAGpos']==1) & (d['NoMej']==0)).sum())
    c = int(((d['BAGpos']==0) & (d['NoMej']==1)).sum())
    d_ = int(((d['BAGpos']==0) & (d['NoMej']==0)).sum())

    # Risks and RR (add 0.5 if any zero cells to avoid infinities)
    a_, b_, c_, d__ = a, b, c, d_
    if 0 in [a,b,c,d_]:
        a_, b_, c_, d__ = a+0.5, b+0.5, c+0.5, d_+0.5

    risk_pos = a_ / (a_ + b_)
    risk_neg = c_ / (c_ + d__)
    RR = risk_pos / risk_neg

    # Katz 95% CI on log scale
    se_log_rr = np.sqrt((1/a_) - (1/(a_+b_)) + (1/c_) - (1/(c_+d__)))
    lo, hi = np.exp(np.log(RR) - 1.96*se_log_rr), np.exp(np.log(RR) + 1.96*se_log_rr)

    # Fisher's exact test (also gives OR). We report its p-value as a quick significance test.
    table = np.array([[a, b],
                      [c, d_]], dtype=int)
    OR_fisher, p_fisher =  stats.fisher_exact(table, alternative='two-sided')

    # Poisson GLM with robust SE to estimate RR (unadjusted; extend formula to adjust)
    glm = smf.glm('NoMej ~ BAGpos', data=d, family=sm.families.Poisson()).fit(cov_type='HC3')
    rr_glm = float(np.exp(glm.params['BAGpos']))
    ci_glm = np.exp(glm.conf_int().loc['BAGpos'].to_numpy())
    p_glm  = float(glm.pvalues['BAGpos'])

    # Nicely print
    n_pos = int((d['BAGpos']==1).sum()); n_neg = int((d['BAGpos']==0).sum())
    print("\n========== Risk of NOT improving by baseline BAG sign ==========")
    print(f"BAG threshold: > {bag_thr:.1f} years considered 'positive'")
    print(f"2×2 (rows=BAGpos [1/0], cols=No mejora [1] / Mejora [0]):")
    print(pd.DataFrame([[a, b],[c, d_]], index=['BAG>thr','BAG≤thr'], columns=['No mejora','Mejora']))
    print(f"\nAbsolute risks:  BAG>thr: {risk_pos:.3f} (n={n_pos})  |  BAG≤thr: {risk_neg:.3f} (n={n_neg})")
    print(f"Relative Risk (Katz): RR={RR:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  |  Fisher p={p_fisher:.4g}")
    print(f"Poisson GLM (robust) RR={rr_glm:.2f}  95% CI [{ci_glm[0]:.2f}, {ci_glm[1]:.2f}]  p={p_glm:.4g}")

    # One-liner suitable for an abstract:
    delta_pct = (RR-1.0)*100
    print("\nAbstract line:")
    print(f"Participants with baseline BAG>{bag_thr:.0f} had a {delta_pct:+.0f}% higher risk of not improving "
          f"(RR={RR:.2f}, 95%CI {lo:.2f}–{hi:.2f}; Fisher p={p_fisher:.3g}; n={len(d)}).")

    return {
        "table": table, "risk_pos": risk_pos, "risk_neg": risk_neg,
        "RR": RR, "RR_CI": (lo, hi), "Fisher_p": float(p_fisher),
        "RR_glm": rr_glm, "RR_glm_CI": (float(ci_glm[0]), float(ci_glm[1])), "RR_glm_p": p_glm,
        "n_total": int(len(d)), "n_BAGpos": n_pos, "n_BAGneg": n_neg
    }


def analyze_bag_vs_freq(freq_cef, wide,
                        use_improve_col=True,
                        mcid_abs=0,
                        mcid_pct=None,
                        save_prefix=None
                       ):
    """
    Requiere en:
      - freq_cef: columnas ['ID','freq_cef_basal','freq_cef_long','Improve'] (Improve=1 mejora; 0 no)
      - wide: columnas ['base_id','BrainPAD_c_t1','BrainPAD_c_t2']

    Pasos:
      1) Merge por ID/base_id (normalizando a mayúsculas)
      2) Calcular ΔBAG y Δfrecuencia
      3) Definir grupo Mejora/No mejora (columna Improve o por umbral MCID)
      4) Spaghetti plot de BAG por grupo
      5) Dispersograma Δfreq vs ΔBAG + correlaciones
      6) Resumen y test de diferencias en ΔBAG entre grupos
      7) (Nuevo) Resumen de BAG (media y sd) en T1 y T2 por grupo
    """
    # --- Preparación / limpieza ---
    fc = freq_cef.copy()
    wd = wide.copy()

    # Asegurar tipos numéricos
    for c in ['freq_cef_basal','freq_cef_long','Improve']:
        if c in fc.columns:
            fc[c] = pd.to_numeric(fc[c], errors='coerce')
    for c in ['BrainPAD_c_t1','BrainPAD_c_t2']:
        wd[c] = pd.to_numeric(wd[c], errors='coerce')

    # Claves para merge
    fc['ID_UP'] = fc['ID'].astype(str).str.upper()
    wd['BASE_UP'] = wd['base_id'].astype(str).str.upper()

    # Merge solo sujetos presentes en ambos
    df = pd.merge(wd, fc, left_on='BASE_UP', right_on='ID_UP', how='inner')

    # --- Cambios ---
    df['dBAG']  = df['BrainPAD_c_t2'] - df['BrainPAD_c_t1']
    df['dFreq'] = df['freq_cef_long'] - df['freq_cef_basal']
    df['dFreq_pct'] = 100.0 * df['dFreq'] / df['freq_cef_basal']

    # --- Grupo Mejora / No mejora ---
    if use_improve_col and 'Improve' in df.columns and df['Improve'].notna().any():
        df['Grupo'] = np.where(df['Improve'] == 1, 'Mejora', 'No mejora')
        fuente_grupo = "columna 'Improve'"
    else:
        if mcid_pct is not None:
            df['Grupo'] = np.where(df['dFreq_pct'] <= mcid_pct, 'Mejora', 'No mejora')
            fuente_grupo = f"umbral porcentual dFreq_pct ≤ {mcid_pct}%"
        else:
            df['Grupo'] = np.where(df['dFreq'] <= -abs(mcid_abs), 'Mejora', 'No mejora')
            fuente_grupo = f"umbral absoluto dFreq ≤ {-abs(mcid_abs)}"

    # --- Chequeo de consistencia (opcional, solo imprime) ---
    if 'Improve' in df.columns and df['Improve'].notna().any():
        neg = (df['dFreq'] < 0).astype(int)
        agree = (neg == df['Improve']).mean()
        print(f"[Info] Acuerdo entre (dFreq<0) e Improve: {agree*100:.1f}% (n={len(df)})")

    # --- Tabla resumen por grupo (Δ) ---
    summary = (df.groupby('Grupo')[['dBAG','dFreq','dFreq_pct']]
                 .agg(['count','mean','std']))
    print("\nResumen por grupo (Δ):\n", summary)

    # --- NUEVO: BAG medio y sd en T1 y T2 por grupo ---
    bag_levels = (df.groupby('Grupo')
                    .agg(n=('BASE_UP','size'),
                         BAG_T1_mean=('BrainPAD_c_t1','mean'),
                         BAG_T1_sd=('BrainPAD_c_t1','std'),
                         BAG_T2_mean=('BrainPAD_c_t2','mean'),
                         BAG_T2_sd=('BrainPAD_c_t2','std')))
    print("\nBAG por grupo (niveles, no Δ):\n", bag_levels.round(3))

    # --- Spaghetti plot (BAG T1/T2) por grupo ---
    long_rows = []
    for _, r in df.iterrows():
        long_rows.append({'ID': r['BASE_UP'], 'tp': 'T1', 'BAG': r['BrainPAD_c_t1'], 'Grupo': r['Grupo']})
        long_rows.append({'ID': r['BASE_UP'], 'tp': 'T2', 'BAG': r['BrainPAD_c_t2'], 'Grupo': r['Grupo']})
    long_df = pd.DataFrame(long_rows)

    fig, ax = plt.subplots(figsize=(7,5))
    for gname, gdf in long_df.groupby('Grupo'):
        # Líneas por sujeto
        for sid, sdf in gdf.groupby('ID'):
            sdf = sdf.sort_values('tp')
            ax.plot([1,2], sdf['BAG'].values, alpha=0.25)
        # Media por tiempo (línea gruesa)
        means = gdf.groupby('tp')['BAG'].mean().reindex(['T1','T2'])
        ax.plot([1,2], means.values, marker='o', linewidth=3, label=gname)
    ax.set_xticks([1,2])
    ax.set_xticklabels(['T1','T2'])
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Brain age gap (corrected)')
    ax.set_title(f'Spaghetti BAG por grupo de cambio clínico ({fuente_grupo})')
    ax.legend(title='Grupo', frameon=False)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_spaghetti.png", dpi=200, bbox_inches='tight')
    plt.show()

    # --- Dispersograma Δfreq vs ΔBAG ---
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(df['dFreq'], df['dBAG'], alpha=0.8)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('Δ Frecuencia cefalea (T2 - T1)')
    ax.set_ylabel('Δ Brain age gap (T2 - T1)')
    ax.set_title('Relación ΔBAG vs ΔFrecuencia')

    # Recta de regresión simple
    if df[['dFreq','dBAG']].dropna().shape[0] >= 2:
        x = df['dFreq'].values
        y = df['dBAG'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            b1, b0 = np.polyfit(x[mask], y[mask], 1)  # y ≈ b1*x + b0
            xs = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xs, b1*xs + b0, linewidth=2)

    # Correlaciones
    if SCIPY and df[['dFreq','dBAG']].dropna().shape[0] >= 3:
        pear = stats.pearsonr(df['dFreq'], df['dBAG'])
        spear = stats.spearmanr(df['dFreq'], df['dBAG'])
        txt = f"Pearson r={pear.statistic:.2f} (p={pear.pvalue:.3f})\nSpearman ρ={spear.correlation:.2f} (p={spear.pvalue:.3f})"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_scatter.png", dpi=200, bbox_inches='tight')
    plt.show()

    # --- Test de diferencias de ΔBAG entre grupos ---
    g = dict(tuple(df.groupby('Grupo')))
    if set(g.keys()) >= {'Mejora','No mejora'}:
        x = g['Mejora']['dBAG'].dropna().values
        y = g['No mejora']['dBAG'].dropna().values

        def hedges_g(a, b):
            na, nb = len(a), len(b)
            sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
            sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2))
            d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
            J = 1 - (3/(4*(na+nb)-9))  # corrección de Hedges
            return d*J

        print("\nComparación ΔBAG (Mejora vs No mejora):")
        print(f"  n Mejora = {len(x)}, mean = {np.mean(x):.2f}, sd = {np.std(x, ddof=1):.2f}")
        print(f"  n NoMej  = {len(y)}, mean = {np.mean(y):.2f}, sd = {np.std(y, ddof=1):.2f}")
        print(f"  Hedges' g = {hedges_g(x,y):.2f}")

        if SCIPY and len(x) >= 2 and len(y) >= 2:
            ttest = stats.ttest_ind(x, y, equal_var=False)  # Welch
            mw = stats.mannwhitneyu(x, y, alternative='two-sided')
            print(f"  Welch t-test: t={ttest.statistic:.2f}, p={ttest.pvalue:.3f}")
            print(f"  Mann–Whitney U: U={mw.statistic:.0f}, p={mw.pvalue:.3f}")

    # Devuelve el dataframe fusionado
    return df

# Function to update BrainPAD
def update_prededad(results_df, predictions_df,
                    id_col_results='ID', id_col_pred='id',
                    pred_col='prediction'):
    r = results_df.copy()
    p = predictions_df.copy()

    # Normaliza tipos/espacios por si acaso
    r[id_col_results] = r[id_col_results].astype(str).str.strip()
    p[id_col_pred]    = p[id_col_pred].astype(str).str.strip()

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

def plot_bag_age_side_by_side(
    df: pd.DataFrame,
    age_col: str = "Edad",
    bag_before_col: str = "BAG",
    bag_after_col: str = "BAG_corr",
    figsize=(10, 5),
    dpi=300,
    outfile: str | None = None,
    annotate: bool = True,
    symmetric_ylim: bool = True,
):
    """
    Create side-by-side plots of BAG vs Age (before vs after correction),
    print correlations/slopes, and return (fig, axes, stats).

    Parameters
    ----------
    df : DataFrame with columns [age_col, bag_before_col, bag_after_col]
    age_col : name of age column (default "Edad")
    bag_before_col : BAG (uncorrected) column (default "BAG")
    bag_after_col : BAG (corrected) column (default "BAG_corr")
    figsize, dpi : figure size and DPI
    outfile : optional path to save the figure (PNG/PDF, etc.)
    annotate : whether to draw a small stats box in each subplot
    symmetric_ylim : if True, uses symmetric y-limits around 0 across both panels

    Returns
    -------
    fig, axes, stats_dict
      stats_dict = {"before": {...}, "after": {...}}
    """

    # --- helper to compute stats and model ---
    def _fit_and_stats(d: pd.DataFrame, y_col: str):
        d = d[[age_col, y_col]].dropna().copy()
        if d.empty:
            raise ValueError(f"No data after dropping NaNs for {age_col} & {y_col}.")
        X = sm.add_constant(d[age_col])
        model = sm.OLS(d[y_col], X).fit()
        slope = float(model.params[age_col])
        p_slope = float(model.pvalues[age_col])

        # Correlations
        r, p_r = stats.pearsonr(d[y_col], d[age_col])
        rho, p_rho = stats.spearmanr(d[y_col], d[age_col])

        stats_out = {
            "n": int(len(d)),
            "Pearson_r": float(r),
        }
        return d, model, stats_out

    # --- fit both ---
    d_before, m_before, s_before = _fit_and_stats(df, bag_before_col)
    d_after,  m_after,  s_after  = _fit_and_stats(df, bag_after_col)

    # Print the correlations/slopes
    def _fmt(s):
        return (f"n={s['n']}, "
                f"Pearson r={s['Pearson_r']:.2f}")

    print("BEFORE (BAG ~ age): " + _fmt(s_before))
    print("AFTER  (BAG ~ age): " + _fmt(s_after))

    # Shared x-range for smoother lines
    xmin = float(min(d_before[age_col].min(), d_after[age_col].min()))
    xmax = float(max(d_before[age_col].max(), d_after[age_col].max()))
    xs = np.linspace(xmin, xmax, 200)
    Xgrid = sm.add_constant(pd.Series(xs, name=age_col))

    y_before_line = m_before.predict(Xgrid)
    y_after_line  = m_after.predict(Xgrid)

    # Shared symmetric y-limits (optional)
    if symmetric_ylim:
        yabs = max(
            np.abs(d_before[bag_before_col]).max(),
            np.abs(d_after[bag_after_col]).max(),
            np.abs(y_before_line).max(),
            np.abs(y_after_line).max(),
        )
        ylim = (-1.05 * yabs, 1.05 * yabs)
    else:
        ylim = None

    # --- plotting ---
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharex=True, sharey=True)

    # BEFORE
    ax = axes[0]
    ax.scatter(d_before[age_col], d_before[bag_before_col], s=18, alpha=0.75)
    ax.plot(xs, y_before_line, linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Brain Age Gap vs Age — Before correction")
    ax.set_xlabel("Age")
    ax.set_ylabel("Brain Age Gap")
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(xmin, xmax)
    if annotate:
        txt = (f"n={s_before['n']}\n"
               f"Pearson r={s_before['Pearson_r']:.2f}\n")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))
    ax.grid(True, alpha=0.25)

    # AFTER
    ax = axes[1]
    ax.scatter(d_after[age_col], d_after[bag_after_col], s=18, alpha=0.75)
    ax.plot(xs, y_after_line, linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Brain Age Gap vs Age — After correction")
    ax.set_xlabel("Age")
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(xmin, xmax)
    if annotate:
        txt = (f"n={s_after['n']}\n"
               f"Pearson r={s_after['Pearson_r']:.2f}\n")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    plt.show()

    stats_dict = {"before": s_before, "after": s_after}
    return fig, axes, stats_dict



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
    out['base_id'] = out['ID'].str.extract(r'(ccov\d+)', expand=False)
    return out

def _parse_ids(df, id_col='ID'):
    out = df.copy()
    # fecha = primeros 8 dígitos (YYYYMMDD) que aparezcan en el ID
    out['fecha'] = pd.to_datetime(out[id_col].str.extract(r'(\d{8})')[0],
                                  format='%Y%m%d', errors='coerce')
    # base_id = ccov + número (sirve para emparejar adquisiciones del mismo sujeto)
    out['base_id'] = out[id_col].str.extract(r'(ccov\d+)')[0]
    # ola = 2/3 si termina en _2/_3; si no, 1 (primera adquisición)
    ola = out[id_col].str.extract(r'_(\d+)$')[0].astype('float')
    out['ola'] = ola.fillna(1).astype(int)
    return out

def emparejar_y_delta(df_base, df_follow, id_col='ID'):
    b = _parse_ids(df_base, id_col)
    f = _parse_ids(df_follow, id_col)

    # Nos quedamos con baseline (ola==1) y segunda adquisición (ola==2)
    b = b[(b['ola'] == 1) & b['base_id'].notna() & b['fecha'].notna()]
    f = f[(f['ola'] == 2) & f['base_id'].notna() & f['fecha'].notna()]

    # Si hubiese duplicados por sujeto, cogemos la más temprana en cada ola
    b = b.sort_values('fecha').drop_duplicates('base_id', keep='first')
    f = f.sort_values('fecha').drop_duplicates('base_id', keep='first')

    # Emparejar por sujeto (base_id) y calcular diferencias
    pares = (b[['base_id', id_col, 'fecha']]
             .merge(f[['base_id', id_col, 'fecha']],
                    on='base_id', suffixes=('_t1', '_t2'), how='inner'))

    pares['delta_dias']  = (pares['fecha_t2'] - pares['fecha_t1']).dt.days
    pares['delta_anos']  = pares['delta_dias'] / 365.25
    return pares.sort_values('base_id').reset_index(drop=True)

def figura_edad_y_edad_predicha(edades_test, pred_test):

    # calculo MAE, MAPE y r test
    MAE_biased_test = mean_absolute_error(edades_test, pred_test)
    r_squared = r2_score(edades_test, pred_test)
    r_biased_test = stats.pearsonr(edades_test, pred_test)[0]

    # Figura concordancia entre predichas y reales con reg lineal
    plt.figure(figsize=(8, 8))
    plt.scatter(edades_test, pred_test, color='blue', label='Predictions')
    plt.plot([10, 100], [10, 100], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Real Age')
    plt.ylabel('Predicted Age')
    plt.title('Predicted Age vs. Real Age')

    # Set x and y axis limits
    plt.xlim(0, 105)
    plt.ylim(0, 105)

    # Ensure x and y axes have the same scale
    plt.gca().set_aspect('equal', adjustable='box')

    # Annotate MAE, Pearson correlation r, and R² in the plot
    textstr = '\n'.join((
        f'MAE: {MAE_biased_test:.2f}',
        f'Pearson r: {r_biased_test:.2f}',
        f'R²: {r_squared:.2f}'))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.show()

def model_evaluation(X_train, X_test, results, features):
    config_parser = configparser.ConfigParser(allow_no_value=True)
    bindir = os.path.abspath(os.path.dirname(__file__))
    config_parser.read(bindir + "/cfg.cnf")

    carpeta_datos = config_parser.get("DATOS", "carpeta_datos")
    carpeta_modelos = config_parser.get("MODELOS", "carpeta_modelos")

    etiv = results['eTIV'].values.tolist()

    X_train = X_train[features]
    X_test = X_test[features]

    X_train, X_test = standardize_data(X_train, X_test)

    # aplico la eliminación de outliers
    X_train, X_test = outlier_flattening_2_entries(X_train, X_test)

    # 3.- normalizo los datos OJO LA NORMALIZACION QUE CON Z NORM O CON 0-1 PUEDE VARIAR EL RESULTADO BASTANTE!
    X_train, X_test = normalize_data_min_max_II(X_train, X_test, (-1, 1))

    file_path = os.path.join(carpeta_modelos, 'ModeloLatest', 'SimpleMLP_nfeats_245_fold_0.pkl')
    with open(file_path, 'rb') as file:
        regresor = pickle.load(file)

    pred_test_median_all = regresor.predict(X_test)
    pred_test_median = pred_test_median_all

    # bias correction
    df_bias_correction = pd.read_csv(os.path.join(carpeta_modelos, 'ModeloLatest', 'DataFrame_bias_correction_1.csv'))

    model = LinearRegression()
    model.fit(df_bias_correction[['edades_train']], df_bias_correction['pred_train'])

    slope = model.coef_[0]
    intercept = model.intercept_

    results['pred_Edad'] = pred_test_median
    results['pred_Edad_c'] = (pred_test_median - intercept) / slope
    results['BrainPAD'] = results['pred_Edad'] - results['Edad']
    results['BrainPAD_c'] = results['pred_Edad_c'] - results['Edad']
    results['eTIV'] = etiv

    fig, axes, stats_ = plot_bag_age_side_by_side(results, age_col="Edad",
                                                   bag_before_col="BrainPAD",
                                                   bag_after_col="BrainPAD_c",
                                                   outfile="BAG_before_after.png")

    ancova_results = pg.ancova(data=results, dv='BrainPAD', between='sexo(M=1;F=0)', covar=['eTIV'])
    print(ancova_results)

    return results

def benjamini_hochberg_correction(p_values):
    n = len(p_values)
    sorted_p_values = np.array(sorted(p_values))
    ranks = np.arange(1, n+1)

    # Calculate the cumulative minimum of the adjusted p-values in reverse
    adjusted_p_values = np.minimum.accumulate((sorted_p_values * n) / ranks)[::-1]

    # Reverse back to original order
    reverse_indices = np.argsort(p_values)
    return adjusted_p_values[reverse_indices]

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, r, r2

config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

carpeta_datos = config_parser.get("DATOS", "carpeta_datos")
carpeta_modelos = config_parser.get("MODELOS", "carpeta_modelos")

controles_COVID = pd.read_csv(os.path.join(carpeta_datos, 'datos_controles', 'Controles_harmo_18_94_FF_noEB_2.csv'))
pac_COVID = pd.read_csv(os.path.join(carpeta_datos, 'datos_controles', 'COVID_I_harmo_18_94_FF_noEB_2.csv'))
pac_COVID_II = pd.read_csv(os.path.join(carpeta_datos, 'datos_controles', 'COVID_II_harmo_18_94_FF_noEB_2.csv'))
X_test_OutSample = pd.read_csv(os.path.join(carpeta_datos, 'datos_controles', 'AgeRisk_harmo_18_94_FF_noEB_2.csv'))
x_test = pd.read_csv(os.path.join(carpeta_datos, 'datos_controles', 'datos_morfo_Harmo_18_94_FF_noEB_2_TEST.csv'))
x_train = pd.read_csv(os.path.join(carpeta_datos, 'datos_controles', 'datos_morfo_Harmo_18_94_FF_noEB_2_TRAIN.csv'))

controles_COVID = controles_COVID[controles_COVID['Escaner'] != 'zarmonitation_1']
pac_COVID = pac_COVID[pac_COVID['Escaner'] != 'zarmonitation_1']
pac_COVID_II = pac_COVID_II[pac_COVID_II['Escaner'] != 'zarmonitation_1']
X_test_OutSample = X_test_OutSample[X_test_OutSample['Escaner'] != 'zarmonitation_1']

controles_COVID = controles_COVID.dropna(how='all')
pac_COVID = pac_COVID.dropna(how='all')
pac_COVID_II = pac_COVID_II.dropna(how='all')
X_test_OutSample = X_test_OutSample.dropna(how='all')

print('Rango edad Controles:')
print(controles_COVID['Edad'].min())
print(controles_COVID['Edad'].max())

print('Rango edad Pacientes:')
print(pac_COVID['Edad'].min())
print(pac_COVID['Edad'].max())

features = pd.read_csv(os.path.join(carpeta_modelos, 'ModeloLatest', 'df_features_con_CoRR.csv'))
features = ast.literal_eval(features.iloc[0, 2])

print(features)

# Evaluo test in sample
test_Age = x_test['Edad'].values
all_data_test = x_test.copy()
X_test_results = x_test.iloc[:, 0:8]
X_test = x_test.iloc[:, 8:]

result_x_test = model_evaluation(x_train, X_test, X_test_results, features)
figura_edad_y_edad_predicha(result_x_test['Edad'].values, result_x_test['pred_Edad_c'].values)

print('######## Resultado test in sample ##########')
res = summarize_metrics(result_x_test, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res)

# Evaluo test out of sample (Age Risk)
test_Age = X_test_OutSample['Edad'].values
all_data_test = X_test_OutSample.copy()
Age_Risk_results = X_test_OutSample.iloc[:, 0:8]
X_test = X_test_OutSample.iloc[:, 8:]

result_AgeRisk = model_evaluation(x_train, X_test, Age_Risk_results, features)
figura_edad_y_edad_predicha(result_AgeRisk['Edad'].values, result_AgeRisk['pred_Edad_c'].values)

print('######## Resultado out of sample ##########')
res = summarize_metrics(result_AgeRisk, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

# aplico la eliminación de outliers
controles_COVID_results = controles_COVID.iloc[:, 0:8]
controles_COVID_results.rename(columns={'sexo': 'sexo(M=1;F=0)'}, inplace=True)
controles_COVID_results = model_evaluation(x_train, controles_COVID, controles_COVID_results, features)
figura_edad_y_edad_predicha(controles_COVID_results['Edad'], controles_COVID_results['pred_Edad'])


print('######## Resultado Controles COVID ##########')
res = summarize_metrics(controles_COVID_results, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

controles_COVID_results.to_csv('controles_COVID_results_morfo.csv', index=False)

# aplico la eliminación de outliers
pac_COVID_results = pac_COVID.iloc[:, 0:8]
pac_COVID_results.rename(columns={'sexo': 'sexo(M=1;F=0)'}, inplace=True)
pac_COVID_results = model_evaluation(x_train, pac_COVID, pac_COVID_results, features)
figura_edad_y_edad_predicha(pac_COVID_results['Edad'], pac_COVID_results['pred_Edad'])

# fragments = ['ccov56', 'ccov74', 'ccov77', '10_ccov7',
# 'ccov12', 'ccov22', 'ccov58', 'ccov70', 'ccov49', 'ccov51', 'ccov52', 'ccov13',
# 'ccov33', 'ccov36', 'ccov47', 'ccov61', 'ccov19', 'ccov68', 'ccov76', '27_ccov1',
# 'ccov63', 'ccov14', 'ccov57', 'ccov46', 'ccov53', 'ccov18', 'ccov25', 'ccov27',
# 'ccov6_2', '_1_ccov5']
#
# pattern = '|'.join(map(re.escape, fragments))
# mask = pac_COVID['ID'].astype(str).str.contains(pattern, case=False, na=False)
# subset = pac_COVID[mask]

# # aplico la eliminación de outliers
# pac_COVID_results = subset.iloc[:, 0:8]
# pac_COVID_results.rename(columns={'sexo': 'sexo(M=1;F=0)'}, inplace=True)
# pac_COVID_results = model_evaluation(x_train, subset, pac_COVID_results, features)
# figura_edad_y_edad_predicha(pac_COVID_results['Edad'], pac_COVID_results['pred_Edad'])

print('######## Resultado Pacientes COVID ##########')
res = summarize_metrics(pac_COVID_results, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

pac_COVID_results.to_csv('pac_COVID_results_morfo.csv', index=False)

# aplico la eliminación de outliers
pac_COVID_II_results = pac_COVID_II.iloc[:, 0:8]
pac_COVID_II_results.rename(columns={'sexo': 'sexo(M=1;F=0)'}, inplace=True)
pac_COVID_II_results = model_evaluation(x_train, pac_COVID_II, pac_COVID_II_results, features)
figura_edad_y_edad_predicha(pac_COVID_II_results['Edad'], pac_COVID_II_results['pred_Edad'])

pac_COVID_results['ID'] = pac_COVID_results['ID'].str.replace(r'(_ccov6)_2$', r'\1', regex=True)
pac_COVID_II_results['ID'] = pac_COVID_II_results['ID'].str.replace(r'(_ccov6)_3$', r'\1_2', regex=True)

pares = emparejar_y_delta(pac_COVID_results, pac_COVID_II_results, id_col='ID')

# IDs baseline que tienen segunda adquisición
ids_long = set(pares['ID_t1'])

# Filtra pac_COVID_results in-place (o crea una copia si prefieres)
pac_COVID_I_results = (pac_COVID_results[pac_COVID_results['ID'].isin(ids_long)].reset_index(drop=True))

print(pac_COVID_I_results.shape)

print('######## Resultado Pacientes COVID t1 ##########')
res = summarize_metrics(pac_COVID_I_results, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

print('######## Resultado Pacientes COVID t2 ##########')
res = summarize_metrics(pac_COVID_II_results, y_col="Edad", yhat_col="pred_Edad", sex_col="sexo(M=1;F=0)", B=5000, seed=42)
print(res.to_string(index=False, max_rows=None, max_cols=None))

pac_COVID_II_results.to_csv('pac_COVID_II_results_morfo.csv', index=False)

controles_COVID_results['Group'] = 'Controls'
pac_COVID_results['Group'] = 'COVID'

prediccionesPymentControls = pd.read_csv('/home/rafa/PycharmProjects/COVID_V2/datos/PrediccionesPyment/pyment_predictions_Controls.csv')
prediccionesPymentCOVID = pd.read_csv('/home/rafa/PycharmProjects/COVID_V2/datos/PrediccionesPyment/pyment_predictions_COVID.csv')
prediccionesPymentCOV_I = pd.read_csv('/home/rafa/PycharmProjects/COVID_V2/datos/PrediccionesPyment/pyment_predictions_COVID_I.csv')
prediccionesPymentCOV_II = pd.read_csv('/home/rafa/PycharmProjects/COVID_V2/datos/PrediccionesPyment/pyment_predictions_COVID_II.csv')

# Apply to all datasets
controles_COVID_results = update_prededad(controles_COVID_results, prediccionesPymentControls)
pac_COVID_results = update_prededad(pac_COVID_results, prediccionesPymentCOVID)
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
for group, name in [(controles_COVID_results, "Controls"), (pac_COVID_results, "MigrCR")]:
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
MigrCR_sex_counts = pac_COVID_results['sexo(M=1;F=0)'].value_counts()

# Create a DataFrame to represent the contingency table
contingency_table = pd.DataFrame({'Healthy': healthy_sex_counts, 'MigrCR': MigrCR_sex_counts,})

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

# Build 2×2 table: rows = sex (M, F), cols = groups (Healthy, MigrCR)
table = np.array([[h[1], p[1]],
                  [h[0], p[0]]], dtype=int)

# Fisher's exact test (two-sided)
oddsratio, p_fisher = stats.fisher_exact(table, alternative='two-sided')
print("\nFisher's Exact Test for Sex:")
print(f"2x2 table (M/F by Healthy/MigrCR):\n{table}")
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
for group, name in [(controles_COVID_results, "Controls"), (pac_COVID_results, "MigrCR")]:
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
print('^^^^^^^^^^^^^^^^^^^^^ MAE r R2 PyBrainAge predictions ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

mae = mean_absolute_error(controles_COVID_results['Edad'], controles_COVID_results['pred_Edad'])
r, _ = pearsonr(controles_COVID_results['Edad'], controles_COVID_results['pred_Edad'])
r2 = r2_score(controles_COVID_results['Edad'], controles_COVID_results['pred_Edad'])

print(f"######################## Healthy group ##############################")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Pearson Correlation Coefficient (r): {r}")
print(f"Coefficient of Determination (R2): {r2}")

mae = mean_absolute_error(pac_COVID_results['Edad'], pac_COVID_results['pred_Edad'])
r, _ = pearsonr(pac_COVID_results['Edad'], pac_COVID_results['pred_Edad'])
r2 = r2_score(pac_COVID_results['Edad'], pac_COVID_results['pred_Edad'])

print(f"######################## MigrCR group ##############################")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Pearson Correlation Coefficient (r): {r}")
print(f"Coefficient of Determination (R2): {r2}")

print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ BRAIN-PAD ANCOVA ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

EN_controls = pd.read_csv('/home/rafa/PycharmProjects/COVID_V2/surface_topology_total_controls.tsv', sep='\t')
EN_pacients = pd.read_csv('/home/rafa/PycharmProjects/COVID_V2/surface_topology_total_COVID_I.tsv', sep='\t')

controles_COVID_results["euler_number"] = controles_COVID_results["ID"].map(EN_controls.set_index("subject")["euler_total"])
pac_COVID_results["euler_number"] = pac_COVID_results["ID"].map(EN_pacients.set_index("subject")["euler_total"])

controles_COVID_results["euler_number"] = controles_COVID_results["euler_number"]/2
pac_COVID_results["euler_number"] = pac_COVID_results["euler_number"]/2

# pick a reference to compute the scaling — controls is a good choice
ref = controles_COVID_results["euler_number"].astype(float)

mu = ref.mean()
sd = ref.std(ddof=0)  # population sd (matches sklearn's StandardScaler)

controles_COVID_results["euler_number_z"] = (controles_COVID_results["euler_number"] - mu) / sd
pac_COVID_results["euler_number_z"] = (pac_COVID_results["euler_number"] - mu) / sd

controles_COVID_results['eTIV'] = controles_COVID_results['eTIV'] / 1000000
pac_COVID_results['eTIV'] = pac_COVID_results['eTIV'] / 1000000

fig, ax = plot_pred_vs_age_two_groups(
    controles_COVID_results,
    pac_COVID_results,
    labels=('Controls', 'COV'),
    outfile_base=None  # e.g. 'age_scatter_nature'
)
plt.show()

merged_df = pd.concat([controles_COVID_results, pac_COVID_results], axis=0)
df = merged_df[['BrainPAD_c', 'BrainPAD', 'sexo(M=1;F=0)', 'Group', 'Edad', 'pred_Edad', 'eTIV', 'euler_number_z']]
df.columns = ['BrainPAD_c', 'BrainPAD', 'sexo', 'Group', 'Age', 'BrainAge', 'eTIV', 'euler_number_z']

df.to_csv('PA_Baseline.scv', index=False)

# === 1) Fit the same ANCOVA with statsmodels OLS to retrieve residuals ===
# BrainPAD ~ Group + covariates
# (C(Group) forces categorical; remove C() if Group is already 0/1 numeric)
df['Group'] = df['Group'].replace({'COVID': 'Patients'})
model = smf.ols('BrainPAD ~ C(Group) + Age + eTIV + sexo + euler_number_z', data=df).fit()

# --- 1) Adjusted group difference (Patients–Controls), CI, p ---
coef_name = [c for c in model.params.index if c.startswith("C(Group)")][0]  # e.g., "C(Group)[T.MigrCR]"
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
            "BrainPAD ~ C(Group) + Age + eTIV + sexo + euler_number_z",
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

pac_COVID_results['ID'] = pac_COVID_results['ID'].str.replace(r'(_ccov6)_2$', r'\1', regex=True)
pac_COVID_II_results['ID'] = pac_COVID_II_results['ID'].str.replace(r'(_ccov6)_3$', r'\1_2', regex=True)

pares = emparejar_y_delta(pac_COVID_results, pac_COVID_II_results, id_col='ID')

# IDs baseline que tienen segunda adquisición
ids_long = set(pares['ID_t1'])

# Filtra pac_COVID_results in-place (o crea una copia si prefieres)
pac_COVID_results = (pac_COVID_results[pac_COVID_results['ID'].isin(ids_long)].reset_index(drop=True))

t1 = add_base_id(pac_COVID_results).copy()
t2 = add_base_id(pac_COVID_II_results).copy()

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

wide.to_csv('PA_Longidtudinal.scv', index=False)

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
pairs = wide[['BrainPAD_c_t1', 'BrainPAD_c_t2', 'BrainPAD_t1', 'BrainPAD_t2']].dropna()
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






