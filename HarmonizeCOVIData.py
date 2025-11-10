from utils_Train import *
import pandas as pd

# ---------- Paths ----------
TRAIN_PATH        = '.../datos_morfo_Harmo_18_94_FF_noEB_2_TRAIN.csv'
AGERISK_PATH      = '.../AgeRisk_noHarmo_18_94_FF_noEB_2.csv'
CONTROLS_PATH    = '.../Controls_COVID.csv'
COVID_I_PATH      = '.../LPI_COVID_I_FastSurfer_V2_data.csv'
COVID_II_PATH     = '.../LPI_COVID_II_FastSurfer_V2_data.csv'

AGERISK_OUT_PATH  = '.../AgeRisk_harmo_18_94_FF_noEB_2.csv'
CONTROLS_OUT_PATH= '.../Controls_harmo_18_94_FF_noEB_2.csv'
COVID_I_OUT_PATH  = '.../COVID_I_harmo_18_94_FF_noEB_2.csv'
COVID_II_OUT_PATH = '.../COVID_II_harmo_18_94_FF_noEB_2.csv'

# ---------- Tags / Names ----------
TAG_TRAIN_SCANNER   = 'zarmonitation_1'          # Ensures reference is 1
TAG_FILTER_SCANNER  = 'zarmonization_1'
ARMO_TAG_AGERISK    = 'Armo_AgeRisk_COVID_18_94_FF_noEB_2'
ARMO_TAG_COVID      = 'Armo_COVID_18_94_FF_noEB_2'
COMBATGAM_DIR       = '.../work/'

# ---------- Load ----------
X_train          = pd.read_csv(TRAIN_PATH)
X_test_OutSample = pd.read_csv(AGERISK_PATH)
Controls        = pd.read_csv(CONTROLS_PATH)
COVID_I          = pd.read_csv(COVID_I_PATH)
COVID_II         = pd.read_csv(COVID_II_PATH)

# ---------- Features ----------
features_to_harmo = X_train.columns.tolist()
print(features_to_harmo)
print(X_train.shape)

if 'eTIV' in features_to_harmo:
    print('eTIV in the dataset')

print(X_train.shape)

# ---------- Reference tag ----------
X_train['Escaner'] = TAG_TRAIN_SCANNER  # Ensures reference is 1

# ---------- Align columns ----------
X_test_OutSample = X_test_OutSample[features_to_harmo]
Controls        = Controls[features_to_harmo]
COVID_I          = COVID_I[features_to_harmo]
COVID_II         = COVID_II[features_to_harmo]

# ---------- Harmonize AgeRisk ----------
datos_armo, my_model = learn_harmonization(X_train, X_test_OutSample, ARMO_TAG_AGERISK)
AgeRisk_harmo = datos_armo[datos_armo['Escaner'] != TAG_FILTER_SCANNER]
AgeRisk_harmo.to_csv(AGERISK_OUT_PATH, index=False)

# ---------- Harmonize COVID (using Controls as healthy set) ----------
datos_armo, my_model = learn_harmonization(X_train, Controls, ARMO_TAG_COVID)
Controls_harmo = datos_armo[datos_armo['Escaner'] != TAG_FILTER_SCANNER]

COVID_I_harmo  = apply_harmonization(COVID_I,  X_train, 1, COMBATGAM_DIR, ARMO_TAG_COVID)
COVID_II_harmo = apply_harmonization(COVID_II, X_train, 1, COMBATGAM_DIR, ARMO_TAG_COVID)

# ---------- Save ----------
Controls_harmo.to_csv(CONTROLS_OUT_PATH, index=False)
COVID_I_harmo.to_csv(COVID_I_OUT_PATH, index=False)
COVID_II_harmo.to_csv(COVID_II_OUT_PATH, index=False)
