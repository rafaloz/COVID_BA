import matplotlib.pyplot as plt
from utils_Harmonization import *
from utils_Train import *
from scipy.stats import pearsonr

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from scipy import stats

X_train = pd.read_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/datos_morfo_Harmo_18_94_FF_noEB_2_TRAIN.csv')
X_test_OutSample = pd.read_csv('/datos/work/rnavgon/ComBatGAM/Models/ModeloCovid/AgeRisk_noHarmo_18_94_FF_noEB_2.csv')

Controles = pd.read_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/Controles_COVID.csv')
COVID_I = pd.read_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/LPI_COVID_I_FastSurfer_V2_data.csv')
COVID_II = pd.read_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/LPI_COVID_II_FastSurfer_V2_data.csv')

features_to_armo = X_train.columns.tolist()

print(features_to_armo)

print(X_train.shape)

if 'eTIV' in features_to_armo:
    print('eTIV dentro')

print(X_train.shape)

X_train['Escaner'] = 'zarmonitation_1'

X_test_OutSample = X_test_OutSample[features_to_armo]
Controles = Controles[features_to_armo]
COVID_I = COVID_I[features_to_armo]
COVID_II = COVID_II[features_to_armo]    

# Armonizo datos AgeRisk
datos_armo, my_model = learn_harmonization(X_train, X_test_OutSample, 'Armo_AgeRisk_COVID_18_94_FF_noEB_2')
AgeRisk_harmo = datos_armo[datos_armo['Escaner'] != 'zarmonization_1']
AgeRisk_harmo.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/AgeRisk_harmo_18_94_FF_noEB_2.csv', index=False)

# Armonizo datos Cardiff_I

datos_armo, my_model = learn_harmonization(X_train, Controles, 'Armo_COVID_18_94_FF_noEB_2')
Controles_harmo = datos_armo[datos_armo['Escaner'] != 'zarmonization_1']
COVID_I_harmo = apply_harmonization(COVID_I, X_train, 1, '/datos/work/rnavgon/ComBatGAM', 'Armo_COVID_18_94_FF_noEB_2')
COVID_II_harmo = apply_harmonization(COVID_II, X_train, 1, '/datos/work/rnavgon/ComBatGAM', 'Armo_COVID_18_94_FF_noEB_2')

Controles_harmo.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/Controles_harmo_18_94_FF_noEB_2.csv', index=False)
COVID_I_harmo.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/COVID_I_harmo_18_94_FF_noEB_2.csv', index=False)
COVID_II_harmo.to_csv('/datos/work/rnavgon/ComBatGAM/DatosCOVID/COVID_II_harmo_18_94_FF_noEB_2.csv', index=False)

print('pausa')

