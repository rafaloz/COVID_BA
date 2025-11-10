from utils_Harmonization import *
from sklearn.model_selection import train_test_split
import pandas as pd

# ---------- Paths ----------
RAW_INPUT_PATH = (".../FastSurfer_data_V2_morfo_sin_armonizar_29_10_2024.csv")

AGERISK_OUT_PATH = (".../AgeRisk_noHarmo_18_94_FF_noEB_2.csv")

TRAIN_OUT_PATH        = ".../datos_morfo_18_94_FF_TRAIN.csv"
VAL_OUT_PATH          = ".../datos_morfo_18_94_FF_VAL.csv"
TEST_OUT_PATH         = ".../datos_morfo_18_94_FF_TEST.csv"

TRAIN_HARMO_OUT_PATH  = ".../datos_morfo_Harmo_18_94_FF_noEB_2_TRAIN.csv"
VAL_HARMO_OUT_PATH    = ".../datos_morfo_Harmo_18_94_FF_noEB_2_VAL.csv"
TEST_HARMO_OUT_PATH   = ".../datos_morfo_Harmo_18_94_FF_noEB_2_TEST.csv"

COMBATGAM_DIR = "/datos/work/rnavgon/ComBatGAM"
ARMO_TAG      = "Armo_COVID_18_94_FF_noEB"

# ---------- Load & initial filters ----------
df = pd.read_csv(RAW_INPUT_PATH)

# Keep adults (>= 18 years)
df = df[df["Edad"] >= 18]

# Split out AgeRisk (kept raw, unharmonized)
agerisk = df[df["DataBase"] == "AgeRisk"]
df = df[df["DataBase"] != "AgeRisk"]
agerisk.to_csv(AGERISK_OUT_PATH, index=False)

# Unique subjects & healthy-only subset
df = df.drop_duplicates(subset=["ID"], keep="first")
df = df[df["Patologia"] == "BD Libre"]

print(df.shape)

# ---------- Scanner pruning ----------
# Keep scanners with at least 30 cases
scanner_counts = df["Escaner"].value_counts()
scanners_to_keep = scanner_counts[scanner_counts >= 30].index

# ---------- Feature selection ----------
# Drop problematic features
problematic_features = [
    # Choroid plexus / ventricles
    "VentricleChoroidVol",
    "Volume_mm3_Left-choroid-plexus",
    "Volume_mm3_Right-choroid-plexus",
    # Hypointensities / vessels
    "Volume_mm3_WM-hypointensities",
    "Volume_mm3_Right-WM-hypointensities",
    "Volume_mm3_Left-WM-hypointensities",
    "Volume_mm3_non-WM-hypointensities",
    "Volume_mm3_Right-non-WM-hypointensities",
    "Volume_mm3_Left-non-WM-hypointensities",
    "Volume_mm3_Left-vessel",
    "Volume_mm3_Right-vessel",
    # Other
    "Volume_mm3_Optic-Chiasm",
    "Volume_mm3_5th-Ventricle",
    "lhSurfaceHoles",
    "rhSurfaceHoles",
    "SurfaceHoles",
    "MaskVol",
    "MaskVol-to-eTIV",
    "BrainSegVol",
    "BrainSegVolNotVent",
    "BrainSegVolNotVentSurf",
    "BrainSegVol-to-eTIV",
    "SupraTentorialVol",
    "SupraTentorialVolNotVent",
    "SupraTentorialVolNotVentVox",
    "CortexVol",
    "TotalGrayVol",
    "CerebralWhiteMatterVol",
    "lhCerebralWhiteMatterVol",
    "rhCerebralWhiteMatterVol",
    "SubCortGrayVol",
    "Volume_mm3_Left_UnsegmentedWhiteMatter",
    "Volume_mm3_Right_UnsegmentedWhiteMatter",
]
df = df.drop(columns=[c for c in problematic_features if c in df.columns])

# Demographics (first 7 columns) + ensure eTIV is present
demographics = df.iloc[:, :7].copy()
if "eTIV" in df.columns and "eTIV" not in demographics.columns:
    demographics["eTIV"] = df["eTIV"]

# Keep morphometrics by name patterns
KEEP_PATTERNS = ("ThickAvg_", "SurfArea_", "GrayVol_", "Volume_mm3")
mask_keep = df.columns.str.contains("|".join(KEEP_PATTERNS))
cols_keep = df.columns[mask_keep]

X = df[cols_keep].copy()
# Drop WM partial-volume volumes
X = X.loc[:, ~X.columns.str.contains("Volume_mm3_wm_", case=False)]

datos_FF = pd.concat([demographics, X], axis=1)

print(datos_FF.columns.tolist())
print(datos_FF.shape)

# ---------- Train/Val/Test split (8:1:1) ----------
# 0) Build _strata (simple or composite batch)
batch_cols = [c for c in ["Escaner", "Bo", "DataBase"] if c in datos_FF.columns]
assert len(batch_cols) >= 1, "At least one batch column (Escaner/Bo/DataBase) is required."
datos_FF["_strata"] = datos_FF[batch_cols].astype(str).agg("|".join, axis=1)

# 1) Avoid classes with a single sample (breaks stratify)
vc = datos_FF["_strata"].value_counts()
datos_FF.loc[datos_FF["_strata"].isin(vc[vc < 2].index), "_strata"] = "OTHER"
use_strat = datos_FF["_strata"].nunique() > 1

# 2) Split 80/10/10 (stratified if multiple classes)
RANDOM_STATE = 42
idx = datos_FF.index

idx_tr, idx_tmp = train_test_split(
    idx, test_size=0.20, random_state=RANDOM_STATE, stratify=datos_FF.loc[idx, "_strata"] if use_strat else None,)
idx_val, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=RANDOM_STATE,
                                   stratify=datos_FF.loc[idx_tmp, "_strata"] if use_strat else None,)

datos_FF = datos_FF.drop(columns=["_strata"])

datos_FF_train = datos_FF.loc[idx_tr].copy()
datos_FF_val   = datos_FF.loc[idx_val].copy()
datos_FF_test  = datos_FF.loc[idx_te].copy()

print(datos_FF_train.columns.tolist())
print(datos_FF_train.shape)

print("Value counts, train:")
print(datos_FF_train["Escaner"].value_counts())
print("Value counts, val:")
print(datos_FF_val["Escaner"].value_counts())
print("Value counts, test:")
print(datos_FF_test["Escaner"].value_counts())

# Save splits
datos_FF_train.to_csv(TRAIN_OUT_PATH, index=False)
datos_FF_val.to_csv(VAL_OUT_PATH, index=False)
datos_FF_test.to_csv(TEST_OUT_PATH, index=False)

# ---------- ComBat-GAM harmonization ----------
# Learn on TRAIN, apply to VAL/TEST
datos_FF_train_harmo, my_model = learn_harmonization_noRef(datos_FF_train, ARMO_TAG)
datos_FF_train_harmo.to_csv(TRAIN_HARMO_OUT_PATH, index=False)

datos_FF_val_harmo = apply_harmonization(datos_FF_val, datos_FF_train, None, COMBATGAM_DIR, ARMO_TAG)
datos_FF_val_harmo.to_csv(VAL_HARMO_OUT_PATH, index=False)

datos_FF_test_harmo = apply_harmonization(datos_FF_test, datos_FF_train, None, COMBATGAM_DIR, ARMO_TAG)
datos_FF_test_harmo.to_csv(TEST_HARMO_OUT_PATH, index=False)
