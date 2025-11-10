
from utils_Harmonization import *
from scipy.stats.mstats import winsorize
from scipy.stats import shapiro, skew, kurtosis
from sklearn.model_selection import train_test_split
import joblib

import numpy as np
import pandas as pd
import seaborn as sns


datos_morfo_noHarmo = pd.read_csv('/datos/work/rnavgon/ComBatGAM/DatosReview/DatosParaAjuste/FastSurfer_data_V2_morfo_sin_armonizar_29_10_2024.csv')
datos_morfo_noHarmo = datos_morfo_noHarmo[datos_morfo_noHarmo['Edad'] >= 18]

AgeRisk = datos_morfo_noHarmo[datos_morfo_noHarmo['DataBase'] == 'AgeRisk']
datos_morfo_noHarmo = datos_morfo_noHarmo[datos_morfo_noHarmo['DataBase'] != 'AgeRisk']

AgeRisk.to_csv('/datos/work/rnavgon/ComBatGAM/Models/ModeloCovid/AgeRisk_noHarmo_18_94_FF_noEB_2.csv', index=False)

datos_morfo_noHarmo = datos_morfo_noHarmo.drop_duplicates(subset=["ID"], keep="first")
datos_morfo_noHarmo = datos_morfo_noHarmo[datos_morfo_noHarmo['Patologia'] == 'BD Libre']

print(datos_morfo_noHarmo.shape)

# Count how many entries each scanner has
# datos_morfo_noHarmo = pd.concat([datos_morfo_noHarmo, AgeRisk], axis=0)
scanner_counts = datos_morfo_noHarmo["Escaner"].value_counts()
scanners_to_keep = scanner_counts[scanner_counts >= 30].index
datos_morfo_noHarmo = datos_morfo_noHarmo[datos_morfo_noHarmo["Escaner"].isin(scanners_to_keep)]
datos_morfo_noHarmo = datos_morfo_noHarmo[datos_morfo_noHarmo["Escaner"]!="IXI_Philips_Medical_Systems_Gyroscan_Intera_1.5T"]
datos_morfo_noHarmo = datos_morfo_noHarmo[datos_morfo_noHarmo["Escaner"]!="CoRR_IPCAS_2_Siemens_TrioTim"]
datos_morfo_noHarmo = datos_morfo_noHarmo[datos_morfo_noHarmo["Escaner"]!="CoRR_LMU_1_Philips_Achieva"]
datos_morfo_noHarmo = datos_morfo_noHarmo[datos_morfo_noHarmo["Escaner"]!="UVA_Philips_Achieva_3T_MRI"]

print(datos_morfo_noHarmo.shape)

problematic_features = [
#     # ----- Choroid Plexus / Ventricles -----
     "VentricleChoroidVol",
     "Volume_mm3_Left-choroid-plexus",
     "Volume_mm3_Right-choroid-plexus",
#
#     # ----- Hypointensities / Vessels -----
     "Volume_mm3_WM-hypointensities",
     "Volume_mm3_Right-WM-hypointensities",
     "Volume_mm3_Left-WM-hypointensities",
     "Volume_mm3_non-WM-hypointensities",
     "Volume_mm3_Right-non-WM-hypointensities",
     "Volume_mm3_Left-non-WM-hypointensities",
     "Volume_mm3_Left-vessel",
     "Volume_mm3_Right-vessel",
#
     'Volume_mm3_Optic-Chiasm',
     'Volume_mm3_5th-Ventricle',

     'lhSurfaceHoles',
     'rhSurfaceHoles',
     'SurfaceHoles',

     'MaskVol',
     'MaskVol-to-eTIV',
     'BrainSegVol',
     'BrainSegVolNotVent',
     'BrainSegVolNotVentSurf',
     'BrainSegVol-to-eTIV',
     'SupraTentorialVol',
     'SupraTentorialVolNotVent',
     'SupraTentorialVolNotVentVox',
     'CortexVol',
     'TotalGrayVol',
     'CerebralWhiteMatterVol',
     'lhCerebralWhiteMatterVol',
     'rhCerebralWhiteMatterVol',
     'SubCortGrayVol', 
     'Volume_mm3_Left_UnsegmentedWhiteMatter', 
     'Volume_mm3_Right_UnsegmentedWhiteMatter'
         ]

datos_morfo_noHarmo = datos_morfo_noHarmo.drop(columns=problematic_features)

demographics = datos_morfo_noHarmo.iloc[:, :7].copy()
demographics['eTIV'] = datos_morfo_noHarmo['eTIV']

KEEP = ('ThickAvg_', 'SurfArea_', 'GrayVol_', 'Volume_mm3')  # pon aquí tus cadenas
mask_keep = datos_morfo_noHarmo.columns.str.contains('|'.join(KEEP))
cols_keep = datos_morfo_noHarmo.columns[mask_keep]
X = datos_morfo_noHarmo[cols_keep].copy()

X = X.loc[:, ~X.columns.str.contains('Volume_mm3_wm_', case=False)]

datos_FF = pd.concat([demographics, X], axis=1)

print(datos_FF.columns.tolist())

print(datos_FF.shape)

# 0) Crear _strata (batch simple o compuesto)
cols = [c for c in ['Escaner','Bo','DataBase'] if c in datos_FF.columns]
assert len(cols) >= 1, "Falta al menos una columna de batch (Escaner/Bo/DataBase)."
datos_FF['_strata'] = datos_FF[cols].astype(str).agg('|'.join, axis=1)

# 1) Evitar clases con un único caso (rompen stratify)
vc = datos_FF['_strata'].value_counts()
datos_FF.loc[datos_FF['_strata'].isin(vc[vc < 2].index), '_strata'] = 'OTHER'
use_strat = datos_FF['_strata'].nunique() > 1

# 2) Tu split 8:1:1 (estratificado si hay más de una clase)
RANDOM_STATE = 42
idx = datos_FF.index

idx_tr, idx_tmp = train_test_split(
    idx, test_size=0.20, random_state=RANDOM_STATE,
    stratify=datos_FF.loc[idx, '_strata'] if use_strat else None
)
idx_val, idx_te = train_test_split(
    idx_tmp, test_size=0.50, random_state=RANDOM_STATE,
    stratify=datos_FF.loc[idx_tmp, '_strata'] if use_strat else None
)

datos_FF = datos_FF.drop(['_strata'], axis=1)

datos_FF_train = datos_FF.loc[idx_tr].copy()
datos_FF_val   = datos_FF.loc[idx_val].copy()
datos_FF_test  = datos_FF.loc[idx_te].copy()

print(datos_FF_train.columns.tolist())

print(datos_FF_train.shape)

print("value counts, train:")
print(datos_FF_train["Escaner"].value_counts())
print("value counts, val:")
print(datos_FF_val["Escaner"].value_counts())
print("value counts, test:")
print(datos_FF_test["Escaner"].value_counts())

datos_FF_train.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/datos_morfo_18_94_FF_TRAIN.csv', index=False)
datos_FF_val.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/datos_morfo_18_94_FF_VAL.csv', index=False)
datos_FF_test.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/datos_morfo_18_94_FF_TEST.csv', index=False)

# 4) Armoniza con ComBat-GAM (learn on TRAIN, apply to VAL/TEST)
datos_FF_train_harmo, my_model = learn_harmonization_noRef(datos_FF_train, 'Armo_COVID_18_94_FF_noEB')
datos_FF_train_harmo.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/datos_morfo_Harmo_18_94_FF_noEB_2_TRAIN.csv', index=False)
datos_FF_val_harmo  = apply_harmonization(datos_FF_val,  datos_FF_train, None, '/datos/work/rnavgon/ComBatGAM', 'Armo_COVID_18_94_FF_noEB')
datos_FF_val_harmo.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/datos_morfo_Harmo_18_94_FF_noEB_2_VAL.csv', index=False)
datos_FF_test_harmo = apply_harmonization(datos_FF_test, datos_FF_train, None, '/datos/work/rnavgon/ComBatGAM', 'Armo_COVID_18_94_FF_noEB')
datos_FF_test_harmo.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/datos_morfo_Harmo_18_94_FF_noEB_2_TEST.csv', index=False)

