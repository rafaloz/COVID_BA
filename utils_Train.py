
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm
from statsmodels.formula.api import glm
import statsmodels.formula.api as smf

from sklearn.linear_model import Lasso

from utils_Harmonization import *

from feature_engine.selection import SmartCorrelatedSelection
from scipy.stats import ks_2samp, levene, ttest_ind, chi2_contingency, kstest, mannwhitneyu
import infoselect as inf

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
import pickle
import os
import copy
from tqdm import tqdm

def split_5_fold_train_test(datos, fold):
    # Assure datos are randomized by rows before entering here
    datos_list = np.array_split(datos, 5)

    datos_test = datos_list[fold]
    # Combine the remaining parts for training data
    remaining_indices = [i for i in range(5) if i != fold]
    datos_train = pd.concat([datos_list[i] for i in remaining_indices], axis=0)

    print('[INFO] Shape de los datos de entrenamiento: ' + str(datos_train.shape))
    print('[INFO] Shape de los datos de prueba: ' + str(datos_test.shape))

    return datos_train, datos_test

def check_split(datos_list, datos_validation, datos_test):

    va_test_check = not(datos_test.equals(datos_validation))
    check_val_train, check_test_train = [], []
    for dataframe in datos_list[2:10]:
        check_val_train.append(not(dataframe.equals(datos_validation)))
        check_test_train.append(not(dataframe.equals(datos_test)))

    return va_test_check and all(check_val_train) and all(check_test_train)


def split_8_1_1(datos, fold):

    # divido los datos en bloques de 10; estńa randomizados por filas antes de entrar aqui
    datos_list = np.array_split(datos, 10)

    datos_test = datos_list[fold]
    if fold == 0:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[2:10]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 1:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[fold+2:10]+[datos_list[fold-1]]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 8:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[0:8]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 9:
        datos_validation = datos_list[0]
        datos_list_check = datos_list[1:9]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    else:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[fold+2:10]+datos_list[0:fold]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')

    print('[INFO] Shape de los datos de entrenamineto ' + str(datos_train.values.shape))

    return datos_train, datos_validation, datos_test

def outlier_flattening(datos_train, datos_val, datos_test):
    datos_train_flat = datos_train.copy()
    datos_val_flat = datos_val.copy()
    datos_test_flat = datos_test.copy()

    for col in datos_train.columns:
        if col == 'sexo':
            continue
        else:
            percentiles = datos_train[col].quantile([0.025, 0.975]).values
            datos_train_flat[col] = np.clip(datos_train[col], percentiles[0], percentiles[1])
            datos_val_flat[col] = np.clip(datos_val[col], percentiles[0], percentiles[1])
            datos_test_flat[col] = np.clip(datos_test[col], percentiles[0], percentiles[1])

    return datos_train_flat, datos_val_flat, datos_test_flat


def outlier_flattening_2_entries(datos_train, datos_test):
    datos_train_flat = datos_train.copy()
    datos_test_flat = datos_test.copy()

    for col in datos_train.columns:
        if col == 'sexo':
            continue
        else:
            percentiles = datos_train[col].quantile([0.025, 0.975]).values
            datos_train_flat[col] = np.clip(datos_train[col], percentiles[0], percentiles[1])
            datos_test_flat[col] = np.clip(datos_test[col], percentiles[0], percentiles[1])

    return datos_train_flat, datos_test_flat


def outlier_flattening_by_3_year_windows_fast(datos_train, edades_train, datos_val, edades_val, datos_test, edades_test):
    # Convert ages to pandas Series if they aren't already
    edades_train = pd.Series(edades_train) if isinstance(edades_train, np.ndarray) else edades_train
    edades_val = pd.Series(edades_val) if isinstance(edades_val, np.ndarray) else edades_val
    edades_test = pd.Series(edades_test) if isinstance(edades_test, np.ndarray) else edades_test

    # Calculate the bin edges for 5-year windows, inclusive of max age in data
    min_age = int(edades_train.min())
    max_age = int(edades_train.max())
    bins = np.arange(min_age, max_age + 4, 3)  # Create bins with a step of 5 years
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]

    # Assign age groups using pd.cut for efficient binning
    datos_train['age_group'] = pd.cut(edades_train, bins=bins, labels=labels, right=False)
    datos_val['age_group'] = pd.cut(edades_val, bins=bins, labels=labels, right=False)
    datos_test['age_group'] = pd.cut(edades_test, bins=bins, labels=labels, right=False)

    # Get numeric columns; this assumes you don't need to recast the types repeatedly if they're already correct
    numeric_cols = datos_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Edad']  # Exclude 'Edad' from outlier clipping

    def calculate_clipping_thresholds(train_df, numeric_cols):
        """ Calculate clipping thresholds based on training data. """
        thresholds = {}
        for col in numeric_cols:
            # Compute percentiles for each group in the training data
            group_percentiles = train_df.groupby('age_group', observed=True)[col].quantile([0.025, 0.975]).unstack()
            thresholds[col] = group_percentiles
        return thresholds

    def clip_outliers(df, numeric_cols, thresholds):
        """ Clip outliers in the dataframe based on provided thresholds from training data. """
        for col in numeric_cols:
            if col in thresholds:
                # Apply clipping using the precomputed thresholds
                df[col] = df.groupby('age_group', observed=True)[col].transform(
                    lambda group: group.clip(lower=thresholds[col].loc[group.name, 0.025],
                                             upper=thresholds[col].loc[group.name, 0.975]))
        return df

    # Assuming 'datos_train', 'datos_val', 'datos_test' are your DataFrames and 'numeric_cols' is your list of numeric columns
    # Calculate thresholds from training data
    thresholds = calculate_clipping_thresholds(datos_train, numeric_cols)

    # Now apply these thresholds to train, validation, and test sets
    datos_train = clip_outliers(datos_train, numeric_cols, thresholds)
    datos_val = clip_outliers(datos_val, numeric_cols, thresholds)
    datos_test = clip_outliers(datos_test, numeric_cols, thresholds)

    # Remove the age_group column as it's no longer needed
    datos_train.drop(columns=['age_group'], inplace=True)
    datos_val.drop(columns=['age_group'], inplace=True)
    datos_test.drop(columns=['age_group'], inplace=True)

    datos_train.loc[:, 'Edad'] = edades_train
    datos_val.loc[:, 'Edad'] = edades_val
    datos_test.loc[:, 'Edad'] = edades_test

    datos_train.to_csv('X_train_out.csv', index=False)
    datos_val.to_csv('X_val_out.csv', index=False)
    datos_test.to_csv('X_test_out.csv', index=False)

    datos_train = datos_train.drop(['Edad'], axis=1)
    datos_val = datos_val.drop(['Edad'], axis=1)
    datos_test = datos_test.drop(['Edad'], axis=1)

    return datos_train, datos_val, datos_test

def outlier_flattening_by_3_year_windows(datos_train, edades_train, datos_test, edades_test):
    datos_train_flat = datos_train.copy()
    datos_test_flat = datos_test.copy()

    # Ensure age data is a pandas Series
    if isinstance(edades_train, np.ndarray):
        edades_train = pd.Series(edades_train)
    if isinstance(edades_test, np.ndarray):
        edades_test = pd.Series(edades_test)

    # Convert all numeric columns to float
    numeric_cols_train = datos_train_flat.select_dtypes(include=[np.number]).columns
    datos_train_flat[numeric_cols_train] = datos_train_flat[numeric_cols_train].astype(float)
    numeric_cols_test = datos_test_flat.select_dtypes(include=[np.number]).columns
    datos_test_flat[numeric_cols_test] = datos_test_flat[numeric_cols_test].astype(float)

    # Create overlapping 3-year age windows
    min_age = int(edades_train.min())
    max_age = int(edades_train.max())

    # Generate age windows
    age_windows = [(start, start + 2) for start in range(min_age, max_age + 1)]
    age_labels = [f"{start}-{end}" for start, end in age_windows]

    # Function to classify age into overlapping windows
    def assign_age_window(age):
        for start, end in age_windows:
            if start <= age <= end:
                return f"{start}-{end}"
        return None

    # Add age group as a column for both train and test data
    datos_train_flat['age_group'] = edades_train.apply(assign_age_window)
    datos_test_flat['age_group'] = edades_test.apply(assign_age_window)

    # Iterate over each age group
    for group in age_labels:
        train_group_indices = datos_train_flat['age_group'] == group
        test_group_indices = datos_test_flat['age_group'] == group

        for col in numeric_cols_train:
            if col in ['Edad']:  # Adjust as necessary for column names
                continue

            percentiles = datos_train_flat.loc[train_group_indices, col].quantile([0.025, 0.975]).values
            datos_train_flat.loc[train_group_indices, col] = np.clip(datos_train_flat.loc[train_group_indices, col], percentiles[0], percentiles[1])
            datos_test_flat.loc[test_group_indices, col] = np.clip(datos_test_flat.loc[test_group_indices, col], percentiles[0], percentiles[1])

    # Remove the age_group column as it's no longer needed
    datos_train_flat.drop(columns=['age_group'], inplace=True)
    datos_test_flat.drop(columns=['age_group'], inplace=True)

    return datos_train_flat, datos_test_flat


def outlier_flattening_by_3_year_bins_2ent(datos_train, edades_train, datos_test, edades_test):
    datos_train_flat = datos_train.copy()
    datos_test_flat = datos_test.copy()

    # Determine the minimum and maximum age in the training data
    min_age = edades_train.min()
    max_age = edades_test.max()

    # Convert all numeric columns to float for training data
    numeric_cols_train = datos_train_flat.select_dtypes(include=[np.number]).columns
    datos_train_flat[numeric_cols_train] = datos_train_flat[numeric_cols_train].astype(float)

    # Convert all numeric columns to float for testing data
    numeric_cols_test = datos_test_flat.select_dtypes(include=[np.number]).columns
    datos_test_flat[numeric_cols_test] = datos_test_flat[numeric_cols_test].astype(float)

    # Create age bins in intervals of 2 years
    age_bins = list(range(int(min_age // 3 * 3), int(max_age // 3 * 3) + 3, 3))
    age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]

    # Add age group as a column for both train and test data
    datos_train_flat['age_group'] = pd.cut(edades_train, bins=age_bins, labels=age_labels, right=False)
    datos_test_flat['age_group'] = pd.cut(edades_test, bins=age_bins, labels=age_labels, right=False)

    # Iterate over each age group
    for group in age_labels:
        # Filter data for the current age group
        train_group_indices = datos_train_flat['age_group'] == group
        test_group_indices = datos_test_flat['age_group'] == group

        # Only select numeric columns
        numeric_cols = datos_train_flat.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col == 'Edad':
                continue

            # Calculate percentiles within this group
            percentiles = datos_train_flat.loc[train_group_indices, col].quantile([0.025, 0.975]).values

            # Apply clipping using .loc
            datos_train_flat.loc[train_group_indices, col] = np.clip(datos_train_flat.loc[train_group_indices, col], percentiles[0], percentiles[1])
            datos_test_flat.loc[test_group_indices, col] = np.clip(datos_test_flat.loc[test_group_indices, col], percentiles[0], percentiles[1])

    # Remove the age_group column since it's not needed anymore
    datos_train_flat.drop(columns=['age_group'], inplace=True)
    datos_test_flat.drop(columns=['age_group'], inplace=True)

    return datos_train_flat, datos_test_flat

def normalize_data_min_max(datos_train, datos_val, datos_test, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_val = scaler.transform(datos_val)
    datos_test = scaler.transform(datos_test)

    # Save the scaler to a file using pickle
    # with open('scaler.pkl', 'wb') as file:
    #     pickle.dump(scaler, file)

    return datos_train, datos_val, datos_test

def normalize_data_min_max_II(datos_train, datos_test, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_test = scaler.transform(datos_test)

    # Save the scaler to a file using pickle
    # with open('scaler.pkl', 'wb') as file:
    #     pickle.dump(scaler, file)

    return datos_train, datos_test


def learn_LS_model(data, covars, ref_batch=None, orig_model=None):
    # transpose data as per ComBat convention
    data = data.T
    # prep covariate data
    covar_levels = list(covars.columns)
    batch_labels = np.unique(covars.SITE)
    batch_col = covars.columns.get_loc('SITE')

    if orig_model is None:
        pass
    else:
        model = copy.deepcopy(orig_model)

    if orig_model is None:
        pass
    else:
        isTrainSite = covars['SITE'].isin(model['SITE_labels'])
        isTrainSiteLabel = set(model['SITE_labels'])
        isTrainSiteColumns = np.where((pd.DataFrame(np.unique(covars['SITE'])).isin(model['SITE_labels']).values).flat)
        isTrainSiteColumnsOrig = np.where((pd.DataFrame(model['SITE_labels']).isin(np.unique(covars['SITE'])).values).flat)
        isTestSiteColumns = np.where((~pd.DataFrame(np.unique(covars['SITE'])).isin(model['SITE_labels']).values).flat)

    cat_cols = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c!='SITE']

    covars = np.array(covars, dtype='object')

    # convert batch col to integer
    if ref_batch is None:
        ref_level = None
    else:
        ref_indices = np.argwhere(covars[:, batch_col] == ref_batch).squeeze()
        if ref_indices.shape[0] == 0:
            ref_level = None
            ref_batch = None
            print('Scanner reference not found. Setting to None.')
            covars[:, batch_col] = np.unique(covars[:, batch_col], return_inverse=True)[-1]
        else:
            covars[:, batch_col] = np.unique(covars[:, batch_col], return_inverse=True)[-1]
            ref_level = covars[ref_indices[0].astype(int), batch_col]

    covars[:, batch_col] = np.unique(covars[:, batch_col], return_inverse=True)[-1]
    # create dictionary that stores batch info
    (batch_levels, sample_per_batch) = np.unique(covars[:, batch_col], return_counts=True)
    n_batch = len(batch_levels)
    batch_info = [list(np.where(covars[:, batch_col] == idx)[0]) for idx in batch_levels]
    info_dict = {
        'batch_levels': batch_levels.astype('int'),
        'ref_level': ref_level,
        'n_batch': len(batch_levels),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch.astype('int'),
        'batch_info': batch_info
    }

    design = make_design_matrix(covars, batch_col, cat_cols, num_cols, ref_level)
    B_hat = np.dot(np.dot(np.linalg.inv(np.dot(design.T, design)), design.T), data.T)
    n_sample = data.shape[1]

    if ref_level is not None:
        grand_mean = np.transpose(B_hat[ref_level, :])
    else:
        grand_mean = np.dot((sample_per_batch / float(n_sample)).T, B_hat[:n_batch, :])
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))

    if ref_level is not None:
        X_ref = data[:, batch_info[ref_level]]
        design_ref = design[batch_info[ref_level],:]
        n_sample_ref = sample_per_batch[ref_level]
        var_pooled = np.dot(((X_ref - np.dot(design_ref, B_hat).T)**2), np.ones((n_sample_ref, 1)) / float(n_sample_ref))
    else:
        var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((n_sample, 1)) / float(n_sample))

    var_pooled[var_pooled == 0] = np.median(var_pooled != 0)

    mod_mean = 0
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:, range(0, n_batch)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))

    model = {'design': design, 'SITE_labels': batch_labels,
             'var_pooled': var_pooled, 'B_hat': B_hat, 'mod_mean': mod_mean, 'grand_mean': grand_mean,
              'info_dict': info_dict, 'SITE_labels_train': batch_labels, 'Covariates': covar_levels}

    return model


def make_design_matrix(Y, batch_col, cat_cols, num_cols, ref_level):
    """
    Return Matrix containing the following parts:
        - one-hot matrix of batch variable (full)
        - one-hot matrix for each categorical_cols (removing the first column)
        - column for each continuous_cols
    """

    def to_categorical(y, nb_classes=None):
        if not nb_classes:
            nb_classes = np.max(y) + 1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y

    hstack_list = []

    ### batch one-hot ###
    # convert batch column to integer in case it's string
    batch = np.unique(Y[:, batch_col], return_inverse=True)[-1]
    if ref_level is not None:
        batch_onehot = to_categorical(batch, len(list(set([np.unique(batch)[0], ref_level]))))
        # batch_onehot[:, ref_level] = np.ones(batch_onehot.shape[0])
    else:
        batch_onehot = to_categorical(batch, len(np.unique(batch)))
    hstack_list.append(batch_onehot)

    ### categorical one-hots ###
    for cat_col in cat_cols:
        cat = np.unique(np.array(Y[:, cat_col]), return_inverse=True)[1]
        cat_onehot = to_categorical(cat, len(np.unique(cat)))[:, 1:]
        hstack_list.append(cat_onehot)

    ### numerical vectors ###
    for num_col in num_cols:
        num = np.array(Y[:, num_col], dtype='float32')
        num = num.reshape(num.shape[0], 1)
        hstack_list.append(num)

    design = np.hstack(hstack_list)
    return design

def apply_LS_model(data, covars, model):

    grand_mean = model['grand_mean']
    mod_mean = model['mod_mean']
    var_pooled = model['var_pooled']
    B_hat = model['B_hat']
    ref_batch = model['info_dict']['ref_level']

    # prep covariate data
    batch_col = covars.columns.get_loc('SITE')
    isTrainSite = covars['SITE'].isin(model['SITE_labels'])
    cat_cols = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c != 'SITE']
    covars = np.array(covars, dtype='object')

    ### additional setup code from neuroCombat implementation:
    # convert training SITEs in batch col to integers
    site_dict = dict(zip(model['SITE_labels'], np.arange(len(model['SITE_labels']))))
    covars[:, batch_col] = np.vectorize(site_dict.get)(covars[:, batch_col], -1)

    # compute samples_per_batch for training data
    sample_per_batch = [np.sum(covars[:, batch_col] == i) for i in list(site_dict.values())]
    sample_per_batch = np.asarray(sample_per_batch)

    # create dictionary that stores batch info
    batch_levels = np.unique(list(site_dict.values()), return_counts=False)
    info_dict = {
        'batch_levels': batch_levels.astype('int'),
        'n_batch': len(batch_levels),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch.astype('int'),
        'batch_info': [list(np.where(covars[:, batch_col] == idx)[0]) for idx in batch_levels],
        'ref_level': ref_batch
    }
    n_batch = len(batch_levels)
    covars[~isTrainSite, batch_col] = 0
    covars[:, batch_col] = covars[:, batch_col].astype(int)

    # isolate array of data in training site
    # apply ComBat without re-learning model parameters
    design = make_design_matrix(covars, batch_col, cat_cols, num_cols, ref_batch)
    design[~isTrainSite, 0:len(model['SITE_labels'])] = np.nan

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, data.shape[1])))

    mod_mean = 0
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:, range(0, n_batch)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))

    # Standardize the data using the calculated mean and variance
    s_data = ((data - stand_mean - mod_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, data.shape[1]))))

    return s_data


def Standardize_LS_2_entries(datos_train, datos_test):

    datos = pd.concat([datos_train, datos_test], axis=0)

    edades = datos['Edad'].values
    maquinas = datos['Escaner'].values
    sex = datos['sexo(M=1;F=0)'].values
    etiv = datos['eTIV'].values

    LE = LabelEncoder()
    datos_maquinas = pd.DataFrame(LE.fit_transform(maquinas))

    d = {'SITE': datos_maquinas.values.ravel().tolist(), 'SEX': np.squeeze(sex).tolist(),
         'ETIV': etiv.tolist(), 'AGE': edades.tolist()}
    covars = pd.DataFrame(data=d)

    datos = datos.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Edad', 'Patologia'], axis=1)
    datos = datos.values

    my_model, datos = harmonizationLearn(datos, covars, eb=False, ref_batch=1)

    return datos


def standardize_data(datos_train, datos_test):
    columns = datos_train.columns

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit to the training data and transform both training and test data
    datos_train_scaled = scaler.fit_transform(datos_train)
    datos_test_scaled = scaler.transform(datos_test)

    # Convert numpy arrays back to pandas DataFrames, retaining original column names and indices
    datos_train_scaled = pd.DataFrame(datos_train_scaled, columns=columns)
    datos_test_scaled = pd.DataFrame(datos_test_scaled, columns=columns)

    return datos_train_scaled, datos_test_scaled


def standardize_data_3entries(datos_train, datos_val, datos_test, features_morphological):
    columns = features_morphological

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit to the training data and transform both training and test data
    datos_train_scaled = scaler.fit_transform(datos_train)
    datos_val_scaled = scaler.transform(datos_val)
    datos_test_scaled = scaler.transform(datos_test)

    # Convert numpy arrays back to pandas DataFrames, retaining original column names and indices
    datos_train_scaled = pd.DataFrame(datos_train_scaled, columns=columns)
    datos_val_scaled = pd.DataFrame(datos_val_scaled, columns=columns)
    datos_test_scaled = pd.DataFrame(datos_test_scaled, columns=columns)

    return datos_train_scaled, datos_val_scaled, datos_test_scaled


def feature_selection(data_train, data_val, data_test, ages_train, n_features):

    # select 10 percent best
    sel_2 = SelectPercentile(mutual_info_regression, percentile=30)
    data_train = sel_2.fit_transform(data_train, ages_train)
    data_val = sel_2.transform(data_val)
    data_test = sel_2.transform(data_test)

    data_train = pd.DataFrame(data_train)
    data_train.columns = sel_2.get_feature_names_out()
    data_val = pd.DataFrame(data_val)
    data_val.columns = sel_2.get_feature_names_out()
    data_test = pd.DataFrame(data_test)
    data_test.columns = sel_2.get_feature_names_out()

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    # Suppress ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Erase correlated feautures
    tr = SmartCorrelatedSelection(variables=None, method="pearson", threshold=0.8, missing_values="raise",
                                  selection_method="model_performance", estimator=MLPRegressor(max_iter=100, early_stopping=True, validation_fraction=0.1),
                                  scoring='neg_mean_absolute_error',  cv=3)

    # dcf = SmartCorrelatedSelection(threshold=0.80, method='pearson', selection_method="variance")
    data_train = tr.fit_transform(data_train, ages_train)
    data_val = tr.transform(data_val)
    data_test = tr.transform(data_test)

    # more MI selection
    gmm = inf.get_gmm(data_train.values, ages_train)
    select = inf.SelectVars(gmm, selection_mode='forward')
    select.fit(data_train.values, ages_train, verbose=False)

    # print(select.get_info())
    # select.plot_mi()
    # select.plot_delta()

    data_train_filtered = select.transform(data_train.values, rd=n_features)
    data_val_filtered = select.transform(data_val.values, rd=n_features)
    data_test_filtered = select.transform(data_test.values, rd=n_features)

    # data_train_filtered = data_train
    # data_test_filtered = data_test
    # features_names = sel_2.get_feature_names_out()

    indices = select.feat_hist[n_features]
    names_list = data_train.columns.tolist()
    features_names = [names_list[i] for i in indices]

    print("nº de features final: "+str(len(features_names)))

    return data_train_filtered, data_val_filtered, data_test_filtered, features_names

def define_lists_cnn():

    # defino listas para guardar los resultados y un dataframe # tab_CNN
    MAE_list_train_tab_CNN, MAE_list_train_unbiased_tab_CNN, r_list_train_tab_CNN, r_list_train_unbiased_tab_CNN, rs_BAG_train_tab_CNN, \
    rs_BAG_train_unbiased_tab_CNN, alfas_tab_CNN, betas_tab_CNN = [], [], [], [], [], [], [], []
    BAG_ChronoAge_df_tab_CNN = pd.DataFrame()

    listas_tab_CNN = [MAE_list_train_tab_CNN, MAE_list_train_unbiased_tab_CNN, r_list_train_tab_CNN,
                  r_list_train_unbiased_tab_CNN, rs_BAG_train_tab_CNN, rs_BAG_train_unbiased_tab_CNN, alfas_tab_CNN,
                  betas_tab_CNN, BAG_ChronoAge_df_tab_CNN, 'tab_CNN']

    return listas_tab_CNN


def execute_in_val_and_test_NN(data_train_filtered, edades_train, data_val_filtered, edades_val, data_test_filtered, edades_test, lista, regresor, n_features, save_dir, fold):

    # identifico en método de regresión
    regresor_used = lista[9]

    # path = '/home/rafa/PycharmProjects/Cardiff_ALSPAC/modelos/modelo_10fcv/fold_8/SimpleMLP_nfeats_100_fold_8.pkl'

    # hago el entrenamiento sobre todos los datos de entrenamiento
    # regresor.fit(data_train_filtered, edades_train, data_val_filtered, edades_val, epochs=500, lr=1e-4, weight_decay=1e-5, batch_size=512, patience=10)
    regresor.fit(data_train_filtered, edades_train, data_val_filtered, edades_val, fold, 245, 16, lr=2*1e-4, weight_decay=1e-3, dropout=0.35, patience=10, batch_size=128)
    # qrf.fit(data_train_filtered, edades_train)

    Elnet = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    Elnet.fit(data_train_filtered, edades_train)

    data_train_filtered = pd.DataFrame(data_train_filtered)
    data_train_filtered['Edades'] = edades_train
    data_val_filtered = pd.DataFrame(data_val_filtered)
    data_val_filtered['Edades'] = edades_val
    data_test_filtered = pd.DataFrame(data_test_filtered)
    data_test_filtered['Edades'] = edades_test

    data_train_filtered.to_csv(os.path.join(save_dir, 'datos_train.csv'), index=False)
    data_val_filtered.to_csv(os.path.join(save_dir, 'datos_val.csv'),  index=False)
    data_test_filtered.to_csv(os.path.join(save_dir, 'datos_test.csv'), index=False)

    edades_train = data_train_filtered['Edades'].values
    data_train_filtered = data_train_filtered.drop('Edades', axis=1)
    data_train_filtered = data_train_filtered.values

    data_val_filtered = data_val_filtered.drop('Edades', axis=1)
    data_val_filtered = data_val_filtered.values

    data_test_filtered = data_test_filtered.drop('Edades', axis=1)
    data_test_filtered = data_test_filtered.values

    # pred_val = regresor.predict(data_val_filtered)['median_deterministic']
    pred_val = regresor.predict(data_val_filtered)

    pred_train_elnet = Elnet.predict(data_val_filtered)

    # Create DataFrame
    df_bias_correction = pd.DataFrame({
        'edades_train': edades_val,
        'pred_train': pred_val
    })

    df_bias_correction.to_csv(os.path.join(save_dir, 'DataFrame_bias_correction_1.csv'), index=False)

    # Create DataFrame
    df_bias_correction = pd.DataFrame({
        'edades_train': edades_val,
        'pred_train': pred_train_elnet
    })

    df_bias_correction.to_csv(os.path.join(save_dir, 'DataFrame_bias_correction_2.csv'), index=False)

    # Hago la predicción de los casos de test sanos
    # pred_test = regresor.predict(data_test_filtered)['median_deterministic']
    pred_test = regresor.predict(data_test_filtered)
    pred_test_all_ridge = Elnet.predict(data_test_filtered)

    # Calculo BAG sanos val & test
    BAG_test_sanos = pred_test - edades_test
    BAG_test_sanos_ridge = pred_test_all_ridge - edades_test

    # calculo MAE, MAPE y r test
    crit = nn.SmoothL1Loss(beta=3)
    Huber_biased_test = crit(torch.tensor(edades_test).float(), torch.tensor(pred_test).float())
    MAE_biased_test = mean_absolute_error(edades_test, pred_test)
    MAPE_biased_test = mean_absolute_percentage_error(edades_test, pred_test)
    r_squared = r2_score(edades_test, pred_test)
    r_biased_test = stats.pearsonr(edades_test, pred_test)[0]
    r_bag_real_biased_test = stats.pearsonr(BAG_test_sanos, edades_test)[0]

    # calculo MAE, MAPE y r test
    MAE_biased_test_ridge = mean_absolute_error(edades_test, pred_test_all_ridge)
    MAPE_biased_test_ridge = mean_absolute_percentage_error(edades_test, pred_test_all_ridge)
    r_squared_ridge = r2_score(edades_test, pred_test_all_ridge)
    r_biased_test_ridge = stats.pearsonr(edades_test, pred_test_all_ridge)[0]
    r_bag_real_biased_test_ridge = stats.pearsonr(BAG_test_sanos_ridge, edades_test)[0]

    # Calculo r MAE para test
    print('----------- ' + regresor_used + ' r & MAE test biased -------------')
    print('MAE test: ' + str(MAE_biased_test))
    print('Huber test: ' + str(Huber_biased_test))
    print('MAPE test: ' + str(MAPE_biased_test))
    print('r test: ' + str(r_biased_test))
    print('R2 test: ' + str(r_squared))

    # calculo r biased test
    print('--------- ' + regresor_used + ' Correlación BAG edad real test -------------')
    print('r BAG-edad real test biased: ' + str(r_bag_real_biased_test))
    print('')

    # Calculo r MAE para test
    print('----------- LASSO r & MAE test biased -------------')
    print('MAE test: ' + str(MAE_biased_test_ridge))
    print('MAPE test: ' + str(MAPE_biased_test_ridge))
    print('r test: ' + str(r_biased_test_ridge))
    print('R2 test: ' + str(r_squared_ridge))

    # calculo r biased test
    print('--------- LASSO Correlación BAG edad real test -------------')
    print('r BAG-edad real test biased: ' + str(r_bag_real_biased_test_ridge))
    print('')


    # Figura concordancia entre predichas y reales con reg lineal
    plt.figure(figsize=(8, 6))
    plt.scatter(edades_test, pred_test, color='blue', label='Predictions')
    plt.plot([edades_test.min(), edades_test.max()], [edades_test.min(), edades_test.max()], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Real Age')
    plt.ylabel('Predicted Age')
    plt.title('Predicted Age vs. Real Age')

    # Annotate MAE, Pearson correlation r, and R² in the plot
    textstr = '\n'.join((
        f'MAE: {MAE_biased_test:.2f}',
        f'Pearson r: {r_biased_test:.2f}',
        f'R²: {r_squared:.2f}'))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # plt.show()
    plt.savefig(save_dir+'/Model_PredAge_vs_Age_fold_'+str(fold)+'.svg')

    MAEs_and_rs_test = pd.DataFrame(list(zip([MAE_biased_test], [r_biased_test], [r_bag_real_biased_test])),
                                    columns=['MAE_biased_test', 'r_biased_test', 'r_bag_real_biased_test'])

    # save the model to disk
    filename = os.path.join(save_dir, 'SimpleMLP_nfeats_' + str(n_features) + '_fold_'+ str(fold) +'.pkl')
    pickle.dump(regresor, open(filename, 'wb'))

    # results = permutation_importance(regresor, data_train_filtered, edades_train, scoring='neg_mean_absolute_error', n_jobs=-1)

    return MAEs_and_rs_test


def plot_ALSPAC_2_grous(df_prev_copy, pval):
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define palette for consistent color use in both violin and scatter plots
    palette = sns.color_palette("pastel", n_colors=df['Group'].nunique())

    # Create a violinplot
    sns.violinplot(x='Group', y='brainPAD_standardized', data=df, inner=None, ax=ax, palette=palette, alpha=1)

    # Create a boxplot, set width of the boxes to 0.1
    sns.boxplot(x='Group', y='brainPAD_standardized', data=df, width=0.1, boxprops={'facecolor': 'None'}, ax=ax, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "10"})

    # Annotate MAE and Median for each group
    groups = df['Group'].unique()
    for i, group in enumerate(groups):
        group_data = df[df['Group'] == group]['brainPAD_standardized']
        mae = np.mean(np.abs(group_data))
        mean = group_data.mean()
        # Position for text annotation can be adjusted as needed
        ax.text(i, group_data.max(), f'MAE: {mae:.2f}\nMean: {mean:.2f}', ha='center', va='bottom')

    # Add a scatterplot for each point, with adjusted color and size
    # Match color with the violin plot and increase size
    sns.stripplot(x='Group', y='brainPAD_standardized', data=df, palette=palette, size=4, jitter=True, ax=ax, edgecolor='gray', linewidth=0.5)

    # Drawing a line between violin plots to show significant difference
    # Adjust these coordinates based on the number of groups and their positions
    x1, x2 = 0, 1  # Columns numbers of your groups
    y, h, col = df['brainPAD_standardized'].max() + 2, 2, 'k'  # y: start of line, h: height of line, col: color
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * .5, y + h, "p-value = "+str(pval), ha='center', va='bottom', color=col)

    # Enhance the plot
    ax.set_title('BrainPAD Values by Group', fontweight='bold')
    ax.set_xlabel('Group', fontweight='bold')
    ax.set_ylabel('BrainPAD Value', fontweight='bold')

    plt.show()


def plot_ALSPAC_4_grous(df):
    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define palette for consistent color use in both violin and scatter plots
    palette = sns.color_palette("pastel", n_colors=df['pliks18TH'].nunique())

    # Create a violinplot with 'pliks30TH' as x and 'BrainPAD' as y
    sns.violinplot(x='pliks18TH', y='brainPAD_standardized', data=df, inner=None, ax=ax, palette=palette)

    # Overlay boxplot
    sns.boxplot(x='pliks18TH', y='brainPAD_standardized', data=df, width=0.1, boxprops={'facecolor': 'None'}, ax=ax, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "10"})

    # Add a scatterplot for each point
    sns.stripplot(x='pliks18TH', y='brainPAD_standardized', data=df, palette=palette, size=4, jitter=True, ax=ax, edgecolor='gray', linewidth=0.5)

    # Calculate medians and plot a trend line
    means = df.groupby('pliks18TH')['brainPAD_standardized'].mean().sort_index()
    plt.plot(means.index, means.values, color="red", marker="o", linestyle="-", linewidth=2, markersize=8,
             label="Median Trend")

    # Calculate and annotate MAE and Median for each 'pliks30TH' category
    categories = df['pliks18TH'].unique()
    for i, category in enumerate(categories):
        category_data = df[df['pliks18TH'] == category]['BrainPAD']
        mae = np.mean(np.abs(category_data))
        mean = category_data.mean()
        # Annotate MAE and Median above each plot
        # Adjust positioning as necessary
        plt.text(i, category_data.max() + 1, f'MAE: {mae:.2f}\nMean: {mean:.2f}', ha='center', va='bottom',
                 fontsize=9, rotation=45)

    # Drawing a line between violin plots to show significant difference
    # Adjust these coordinates based on the number of groups and their positions
    x1, x4 = 0, 3  # Columns numbers of your groups
    y, h, col = df['brainPAD_standardized'].max() + 2, 2, 'k'  # y: start of line, h: height of line, col: color
    ax.plot([x1, x1, x4, x4], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x4) * .5, y + h, "p-value = 0.296", ha='center', va='bottom', color=col)

    # Enhance the plot
    ax.set_title('BrainPAD Values by pliks18TH Categories', fontweight='bold')
    ax.set_xlabel('pliks18TH Categories', fontweight='bold')
    ax.set_ylabel('BrainPAD Value', fontweight='bold')

    plt.show()

def select_controls(patient_group, healthy_group):
    # Determine age range based on the pathology
    patologia = patient_group['Patologia'].unique().item()  # Assuming there's exactly one unique pathology per group
    if patologia == 'Bipolar':
        age_min, age_max = 18, 65
    elif patologia in ['SQZFR_cron', 'SQZFR_primer_ep']:
        age_min, age_max = 18, 60
    else:
        raise ValueError("Unsupported pathology")

    # Filter healthy group based on age
    filtered_healthy_group = healthy_group[(healthy_group['Edad'] >= age_min) & (healthy_group['Edad'] <= age_max)]

    # Determine size of the control group with random variation
    random_change = random.randint(-7, 7)
    group_size = patient_group.shape[0] + random_change

    # Ensure there are enough healthy individuals to sample from
    if filtered_healthy_group.shape[0] < group_size:
        raise ValueError("Not enough healthy controls available")

    # Sample the control group
    control_group = filtered_healthy_group.sample(n=group_size, replace=False)
    return control_group


def check_group(groups):
    controls = {}
    used_controls = pd.DataFrame()

    ks_results = {}
    chi_squared_results = {}

    for group_name in ['Bipolar', 'SQZFR_cron', 'SQZFR_primer_ep']:
        if group_name not in groups:
            continue

        for i in range(0, 100000, 1):

            control_group = select_controls(groups[group_name], groups['Controles'])
            controls[f'control_for_{group_name}'] = control_group

            # Testing group1 for normality
            ks_stat1, ks_p1 = kstest(groups[group_name]['Edad'], 'norm', args=(np.mean(groups[group_name]['Edad']), np.std(groups[group_name]['Edad'])))
            print(f"Group 1 Normality KS test: Statistic={ks_stat1}, P-value={ks_p1}")

            # Testing group2 for normality
            ks_stat2, ks_p2 = kstest(control_group['Edad'], 'norm', args=(np.mean(control_group['Edad']), np.std(control_group['Edad'])))
            print(f"Group 2 Normality KS test: Statistic={ks_stat2}, P-value={ks_p2}")

            lev_stat, lev_p = levene(groups[group_name]['Edad'], control_group['Edad'])
            print(f"Levene’s test: Statistic={lev_stat}, P-value={lev_p}")

            # Use Welch's t-test if variances are not equal (Levene’s test p-value < 0.05), otherwise use standard t-test.
            if lev_p < 0.05:
                t_stat, t_p = ttest_ind(groups[group_name]['Edad'], control_group['Edad'], equal_var=False)
                print("Using Welch's t-test")
            else:
                t_stat, t_p = ttest_ind(groups[group_name]['Edad'], control_group['Edad'], equal_var=True)
                print("Using standard t-test")

            print(f"T-test: Statistic={t_stat}, P-value={t_p}")

            # Perform the Chi-squared test for sex distribution
            table_sex = pd.DataFrame({'Diseased': groups[group_name]['sexo(M=1;F=0)'].value_counts(),
                                      'Control': control_group['sexo(M=1;F=0)'].value_counts()})
            chi2_stat, chi2_p, _, _ = chi2_contingency(table_sex)
            chi_squared_results[group_name] = (chi2_stat, chi2_p)

            print(f"Results for {i}:")
            print(f"p-value for age in control_for_{group_name}: {t_p:.4f}")
            print(f"p-value for sex in control_for_{group_name}: {chi2_p:.4f}")

            if group_name == 'Bipolar':
                if t_p > 0.15 and chi2_p > 0.15:
                    break
            elif group_name == 'SQZFR_cron':
                if t_p > 0.3 and chi2_p > 0.3:
                    break
            elif group_name == 'SQZFR_primer_ep':
                if t_p > 0.4 and chi2_p > 0.4:
                    break

    return controls


def standarization_with_age(X_train, age_column_train, X_val, age_column_val, X_test, age_column_test):

    features = X_train.columns.tolist()

    # Combine intercept and age into the design matrix
    age_column_train = age_column_train.reshape(np.shape(age_column_train)[0], 1)
    age_column_val = age_column_val.reshape(np.shape(age_column_val)[0], 1)
    age_column_test = age_column_test.reshape(np.shape(age_column_test)[0], 1)

    # X.T is the transpose of your data matrix if features are rows and samples are columns
    intercept = np.ones((age_column_train.shape[0], 1))
    design_train = np.hstack([intercept, age_column_train])
    B_hat = np.linalg.inv(design_train.T @ design_train) @ (design_train.T @ X_train) # Your Beta

    intercept = np.ones((age_column_val.shape[0], 1))
    design_val = np.hstack([intercept, age_column_val])
    intercept = np.ones((age_column_test.shape[0], 1))
    design_test = np.hstack([intercept, age_column_test])

    # Compute model-predicted values (mod_mean)
    mod_mean_train = design_train @ B_hat
    mod_mean_val = design_val @ B_hat
    mod_mean_test = design_test @ B_hat

    # Compute the variance (pooled across all samples if considering the whole dataset as one "batch")
    var_pooled = np.mean((X_train - mod_mean_train)**2, axis=0)
    var_pooled[var_pooled == 0] = np.median(var_pooled[var_pooled != 0])  # Handle zero variance

    # Standardize the data
    X_train = (X_train.values - mod_mean_train.values) / np.sqrt(var_pooled.values)
    X_val = (X_val.values - mod_mean_val.values) / np.sqrt(var_pooled.values)
    X_test = (X_test.values - mod_mean_test.values) / np.sqrt(var_pooled.values)

    model = {'B_hat': B_hat, 'std': np.sqrt(var_pooled)}

    X_train = pd.DataFrame(X_train, columns=features)
    X_val = pd.DataFrame(X_val, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)

    return X_train, X_val, X_test

def standarization_with_age(X_train, age_column_train, X_test, age_column_test):

    features = X_train.columns.tolist()

    # Combine intercept and age into the design matrix
    age_column_train = age_column_train.reshape(np.shape(age_column_train)[0], 1)
    age_column_test = age_column_test.reshape(np.shape(age_column_test)[0], 1)

    # X.T is the transpose of your data matrix if features are rows and samples are columns
    intercept = np.ones((age_column_train.shape[0], 1))
    design_train = np.hstack([intercept, age_column_train])
    B_hat = np.linalg.inv(design_train.T @ design_train) @ (design_train.T @ X_train) # Your Beta

    intercept = np.ones((age_column_test.shape[0], 1))
    design_test = np.hstack([intercept, age_column_test])

    # Compute model-predicted values (mod_mean)
    mod_mean_train = design_train @ B_hat
    mod_mean_test = design_test @ B_hat

    # Compute the variance (pooled across all samples if considering the whole dataset as one "batch")
    var_pooled_train = np.mean((X_train - mod_mean_train)**2, axis=0)
    var_pooled_train[var_pooled_train == 0] = np.median(var_pooled_train[var_pooled_train != 0])  # Handle zero variance

    # Compute the variance (pooled across all samples if considering the whole dataset as one "batch")
    var_pooled_test = np.mean((X_test - mod_mean_test)**2, axis=0)
    var_pooled_test[var_pooled_test == 0] = np.median(var_pooled_test[var_pooled_test != 0])  # Handle zero variance

    # Standardize the data
    X_train = (X_train.values - mod_mean_train.values) / np.sqrt(var_pooled_train.values)
    X_test = (X_test.values - mod_mean_test.values) / np.sqrt(var_pooled_test.values)

    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)

    return X_train, X_test


def standarization_erase_age(X_train, age_column_train, X_test, age_column_test):
    features = X_train.columns.tolist()

    age_column_train = age_column_train.reshape(np.shape(age_column_train)[0], 1)
    age_column_test = age_column_test.reshape(np.shape(age_column_test)[0], 1)

    # Initialize the intercepts
    intercept_train = np.ones((age_column_train.shape[0], 1))
    intercept_test = np.ones((age_column_test.shape[0], 1))

    # Combine intercept and age into the design matrices
    design_train = np.hstack([intercept_train, age_column_train])
    design_test = np.hstack([intercept_test, age_column_test])

    # Calculate B_hat using the training data
    B_hat = np.linalg.inv(design_train.T @ design_train) @ (design_train.T @ X_train.values)

    # Compute model-predicted values (mod_mean)
    mod_mean_train = design_train @ B_hat
    mod_mean_test = design_test @ B_hat

    # Compute the variance (pooled across all training samples)
    var_pooled_train = np.mean((X_train.values - mod_mean_train)**2, axis=0)
    var_pooled_train[var_pooled_train == 0] = np.median(var_pooled_train[var_pooled_train != 0])  # Handle zero variance

    # Standardize the data
    X_train_standardized = (X_train.values - mod_mean_train) / np.sqrt(var_pooled_train)
    X_test_standardized = (X_test.values - mod_mean_test) / np.sqrt(var_pooled_train)

    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_standardized, columns=features)
    X_test = pd.DataFrame(X_test_standardized, columns=features)

    return X_train, X_test



def outlier_flattening_by_3_year_windows_fast(datos_train, edades_train, datos_val, edades_val, datos_test, edades_test):
    datos_train_flat = datos_train.copy()
    datos_val_flat = datos_val.copy()
    datos_test_flat = datos_test.copy()

    # Ensure age data is a pandas Series
    if isinstance(edades_train, np.ndarray):
        edades_train = pd.Series(edades_train)
    if isinstance(edades_val, np.ndarray):
        edades_val = pd.Series(edades_val)
    if isinstance(edades_test, np.ndarray):
        edades_test = pd.Series(edades_test)

    # Convert all numeric columns to float
    numeric_cols_train = datos_train_flat.select_dtypes(include=[np.number]).columns
    datos_train_flat[numeric_cols_train] = datos_train_flat[numeric_cols_train].astype(float)
    numeric_cols_val = datos_val_flat.select_dtypes(include=[np.number]).columns
    datos_val_flat[numeric_cols_val] = datos_val_flat[numeric_cols_val].astype(float)
    numeric_cols_test = datos_test_flat.select_dtypes(include=[np.number]).columns
    datos_test_flat[numeric_cols_test] = datos_test_flat[numeric_cols_test].astype(float)

    # Create overlapping 3-year age windows
    min_age = int(edades_train.min())
    max_age = int(edades_train.max())

    # Function to classify age into overlapping windows
    def assign_age_window(age):
        for start, end in age_windows:
            if start <= age <= end:
                return f"{start}-{end}"
        return None

    # Generate age windows
    age_windows = [(start, start + 2) for start in range(min_age, max_age + 1)]
    age_labels = [f"{start}-{end}" for start, end in age_windows]

    # Assuming edades_train, edades_val, edades_test are Series or similar
    # Compute age groups
    age_group_train = edades_train.apply(assign_age_window).rename('age_group')
    age_group_val = edades_val.apply(assign_age_window).rename('age_group')
    age_group_test = edades_test.apply(assign_age_window).rename('age_group')

    # Concatenate the new columns to the original DataFrames (Evitar warning the fragmentación de los datos)
    datos_train_flat = pd.concat([datos_train_flat, age_group_train], axis=1)
    datos_val_flat = pd.concat([datos_val_flat, age_group_val], axis=1)
    datos_test_flat = pd.concat([datos_test_flat, age_group_test], axis=1)

    # Precompute indices for each group in each dataset to avoid recalculating
    group_indices_train = {group: datos_train_flat['age_group'] == group for group in age_labels}
    group_indices_val = {group: datos_val_flat['age_group'] == group for group in age_labels}
    group_indices_test = {group: datos_test_flat['age_group'] == group for group in age_labels}

    # Compute quantiles using training data
    def compute_quantiles(dataframe, group_indices):
        quantiles = {}
        for group, indices in group_indices.items():
            group_quantiles = {}
            for col in numeric_cols_train:
                if col == 'Edad':  # Skip specific columns as needed
                    continue

                # Compute the 2.5th and 97.5th percentiles for the column within the current group
                group_quantiles[col] = dataframe.loc[indices, col].quantile([0.025, 0.975]).values
            quantiles[group] = group_quantiles
        return quantiles

    # Function to apply clipping based on precomputed quantiles
    def apply_clipping(dataframe, group_indices, quantiles):
        for group, indices in group_indices.items():
            for col, percentile_values in quantiles[group].items():
                if col not in dataframe.columns:
                    continue  # Skip if the column isn't in the current dataframe
                dataframe.loc[indices, col] = np.clip(dataframe.loc[indices, col], percentile_values[0],
                                                      percentile_values[1])

    # Precompute quantiles using the training data
    quantiles_train = compute_quantiles(datos_train_flat, group_indices_train)

    # Apply clipping to each dataset using the training quantiles as reference
    apply_clipping(datos_train_flat, group_indices_train, quantiles_train)
    apply_clipping(datos_val_flat, group_indices_val, quantiles_train)
    apply_clipping(datos_test_flat, group_indices_test, quantiles_train)

    # Remove the age_group column as it's no longer needed
    datos_train_flat.drop(columns=['age_group'], inplace=True)
    datos_val_flat.drop(columns=['age_group'], inplace=True)
    datos_test_flat.drop(columns=['age_group'], inplace=True)

    return datos_train_flat, datos_val_flat, datos_test_flat


def outlier_flattening_by_3_year_windows_fast(datos_train, edades_train, datos_test, edades_test):
    datos_train_flat = datos_train.copy()
    datos_test_flat = datos_test.copy()

    # Ensure age data is a pandas Series
    if isinstance(edades_train, np.ndarray):
        edades_train = pd.Series(edades_train)
    if isinstance(edades_test, np.ndarray):
        edades_test = pd.Series(edades_test)

    # Convert all numeric columns to float
    numeric_cols_train = datos_train_flat.select_dtypes(include=[np.number]).columns
    datos_train_flat[numeric_cols_train] = datos_train_flat[numeric_cols_train].astype(float)
    numeric_cols_test = datos_test_flat.select_dtypes(include=[np.number]).columns
    datos_test_flat[numeric_cols_test] = datos_test_flat[numeric_cols_test].astype(float)

    # Create overlapping 3-year age windows
    min_age = int(edades_train.min())
    max_age = int(edades_train.max())

    # Function to classify age into overlapping windows
    def assign_age_window(age):
        for start, end in age_windows:
            if start <= age <= end:
                return f"{start}-{end}"
        return None

    # Generate age windows
    age_windows = [(start, start + 2) for start in range(min_age, max_age + 1)]
    age_labels = [f"{start}-{end}" for start, end in age_windows]

    # Assuming edades_train, edades_val, edades_test are Series or similar
    # Compute age groups
    age_group_train = edades_train.apply(assign_age_window).rename('age_group')
    age_group_test = edades_test.apply(assign_age_window).rename('age_group')

    # Concatenate the new columns to the original DataFrames (Evitar warning the fragmentación de los datos)
    datos_train_flat = pd.concat([datos_train_flat, age_group_train], axis=1)
    datos_test_flat = pd.concat([datos_test_flat, age_group_test], axis=1)

    # Precompute indices for each group in each dataset to avoid recalculating
    group_indices_train = {group: datos_train_flat['age_group'] == group for group in age_labels}
    group_indices_test = {group: datos_test_flat['age_group'] == group for group in age_labels}

    # Compute quantiles using training data
    def compute_quantiles(dataframe, group_indices):
        quantiles = {}
        for group, indices in group_indices.items():
            group_quantiles = {}
            for col in numeric_cols_train:
                if col == 'Edad':  # Skip specific columns as needed
                    continue

                # Compute the 2.5th and 97.5th percentiles for the column within the current group
                group_quantiles[col] = dataframe.loc[indices, col].quantile([0.10, 0.90]).values
            quantiles[group] = group_quantiles
        return quantiles

    # Function to apply clipping based on precomputed quantiles
    def apply_clipping(dataframe, group_indices, quantiles):
        for group, indices in group_indices.items():
            for col, percentile_values in quantiles[group].items():
                if col not in dataframe.columns:
                    continue  # Skip if the column isn't in the current dataframe
                dataframe.loc[indices, col] = np.clip(dataframe.loc[indices, col], percentile_values[0],
                                                      percentile_values[1])

    # Precompute quantiles using the training data
    quantiles_train = compute_quantiles(datos_train_flat, group_indices_train)

    # Apply clipping to each dataset using the training quantiles as reference
    apply_clipping(datos_train_flat, group_indices_train, quantiles_train)
    apply_clipping(datos_test_flat, group_indices_test, quantiles_train)

    # Remove the age_group column as it's no longer needed
    datos_train_flat.drop(columns=['age_group'], inplace=True)
    datos_test_flat.drop(columns=['age_group'], inplace=True)

    return (datos_train_flat, datos_test_flat)


def plot_ALSPAC_2_grous_sex(df_prev_copy, pval):
    df = df_prev_copy.copy()

    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define palette for consistent color use in both violin and scatter plots
    palette = sns.color_palette("pastel", n_colors=df['sexo'].nunique())

    # Create a violinplot
    sns.violinplot(x='sexo', y='BrainPAD', data=df, inner=None, ax=ax, palette=palette, alpha=0.2)

    # Create a boxplot, set width of the boxes to 0.1
    sns.boxplot(x='sexo', y='BrainPAD', data=df, width=0.1, boxprops={'facecolor': 'None'}, ax=ax, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "10"})

    # Annotate MAE and Median for each group
    groups = np.sort(df['sexo'].unique())
    for i, group in enumerate(groups):
        group_data = df[df['sexo'] == group]['BrainPAD']
        mae = group_data.abs().mean()
        mean = group_data.mean()
        # Position for text annotation can be adjusted as needed
        ax.text(i, group_data.max() + 1, f'MAE: {mae:.2f}\nMean: {mean:.2f}', ha='center', va='bottom')

    # Add a scatterplot for each point, with adjusted color and size
    # Match color with the violin plot and increase size
    sns.stripplot(x='sexo', y='BrainPAD', data=df, palette=palette, size=4, jitter=True, dodge=True, ax=ax,
                  edgecolor='gray', linewidth=0.5)

    # Drawing a line between violin plots to show significant difference
    # Adjust these coordinates based on the number of groups and their positions
    x1, x2 = 0, 1  # Columns numbers of your groups
    y, h, col = df['BrainPAD'].max() + 2, 2, 'k'  # y: start of line, h: height of line, col: color
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * .5, y + h, "p-value = "+str(pval), ha='center', va='bottom', color=col)

    # Enhance the plot
    ax.set_title('BrainPAD Values by sexo', fontweight='bold')
    ax.set_xlabel('sexo', fontweight='bold')
    ax.set_ylabel('BrainPAD Value', fontweight='bold')

    plt.show()


# Función para ajustar el GLM y devolver el coeficiente de interés
def fit_glm_and_get_coef(data, formula, coef_name):
    model = glm(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    return model.params[coef_name]

# Función para ajustar el GLM y devolver todos los coefs
def fit_glm_and_get_all_coef(data, formula):
    model = glm(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    return model.params

# Define a function to perform the t-test, Welch's test, or U Mann-Whitney test
def perform_tests(group0, group1):
    stat, p_value = ttest_ind(group0, group1)
    return stat

def rain_cloud_plot(df_prev_copy):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['Group'].unique()

    for group in groups:
        x_values[group] = df[df['Group'] == group]['BrainPAD'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    start_blue_colors = '#91bec8'
    end_blue_colors = '#0e4e6e'

    # Generate gradient colors for different groups
    start_color_rgb = hex_to_rgb(start_blue_colors)
    end_color_rgb = hex_to_rgb(end_blue_colors)
    palette_1 = interpolate_colors(start_color_rgb, end_color_rgb, len(groups))

    start_blue_colors = '#6fa4b7'
    end_blue_colors = '#146e9b'
    palette_2 = interpolate_colors(hex_to_rgb(start_blue_colors), hex_to_rgb(end_blue_colors), len(groups))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with axes swapped
    violins = sns.violinplot(data=df, y='BrainPAD', x='Group', hue='Group', palette=palette_2, ax=ax, fill=False,
                             inner_kws=dict(  # Custom settings for the inner boxplot
                                 box_width=0.2,  # Custom width for the inner boxplot
                                 whis_width=2,  # Custom whisker width
                                 color="black",  # Custom color for the inner boxplot and whiskers
                             ),
                             legend=False)

    # Apply alpha to the violin plot elements
    for violin in violins.collections:
        violin.set_alpha(0.3)


    # Scatter plot on top of the violin plot with axes swapped
    for x, val, c in zip(xs, vals, palette_1):
        plt.scatter(x, val, alpha=0.4, color=c, edgecolor=c, zorder=1)

    # Customizing plot
    plt.ylabel('BrainPAD', fontweight='bold')
    plt.xlabel('Group', fontweight='bold')
    plt.title('Raincloud Plot with Violin for BrainPAD by Group (Axes Switched)', fontweight='bold')

    plt.show()


def rain_cloud_plot_II(df_prev_copy):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['Group'].unique()

    for group in groups:
        x_values[group] = df[df['Group'] == group]['BrainPAD'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    start_blue_colors = '#91bec8'
    end_blue_colors = '#0e4e6e'

    # Generate gradient colors for different groups
    start_color_rgb = hex_to_rgb(start_blue_colors)
    end_color_rgb = hex_to_rgb(end_blue_colors)
    palette_1 = interpolate_colors(start_color_rgb, end_color_rgb, len(groups))

    start_blue_colors = '#6fa4b7'
    end_blue_colors = '#146e9b'
    palette_2 = interpolate_colors(hex_to_rgb(start_blue_colors), hex_to_rgb(end_blue_colors), len(groups))

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with no inner fill but outlined box
    sns.violinplot(
        data=df,
        x='BrainPAD',
        y='Group',
        inner=None,  # No inner plots as we will overlay the boxplot
        palette=palette_2,  # Color palette for violins
        linewidth=1.5,  # Line width for the violin outline
        ax=ax
    )

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    sns.boxplot(
        data=df,
        x='BrainPAD',
        y='Group',
        width=0.2,  # Narrower box width
        boxprops=dict(facecolor='none', edgecolor=palette_2, linewidth=1),  # Only outline for the box, no fill
        whiskerprops=dict(linewidth=1),  # Thicker whiskers
        capprops=dict(linewidth=1),  # Thicker caps
        medianprops=dict(linewidth=1, color=palette_2),  # Thicker median line
        showfliers=False,  # Optional: Hide outliers
        ax=ax
    )

    # Customizing plot
    plt.ylabel('Group', fontweight='bold')
    plt.xlabel('BrainPAD', fontweight='bold')
    plt.title('Violin Plot with Outline Boxplot for BrainPAD by Group', fontweight='bold')

    plt.show()


def rain_cloud_plot_III(df_prev_copy):
    import matplotlib
    print("Available backends:", matplotlib.rcsetup.all_backends)
    print("Current backend:", matplotlib.get_backend())
    matplotlib.use('svg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    df = df_prev_copy

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['Group'].unique()

    for group in groups:
        x_values[group] = df[df['Group'] == group]['brainPAD_standardized'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    grey_colors_light = '#B4BBBB'
    gray_colors_dark = '#858E8D'
    blue_colors_light = '#80AFEC'
    blue_colors_dark = '#1D5FB5'
    red_colors_light = '#BF6673'
    red_colors_dark = '#C42840'
    orange_colors_light = '#EC9468'
    orange_colors_dark = '#DC5614'

    # Generate gradient colors for different groups
    palette_1 = interpolate_colors(hex_to_rgb(grey_colors_light), hex_to_rgb(red_colors_light), len(groups))

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['Group'] = df['Group'].replace('Crazy', 'PE_1-3')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with no fill and switched axes (Group on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='Group',
        y='brainPAD_standardized',
        palette=[gray_colors_dark, red_colors_dark],  # Color palette for violin outlines
        linewidth=3.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(groups):
        sns.boxplot(
            data=df[df['Group'] == group],
            x='Group',
            y='brainPAD_standardized',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='Group',
        y='brainPAD_standardized',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=[grey_colors_light, red_colors_light],  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(groups):
        group_mean = df[df['Group'] == group]['brainPAD_standardized'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('Group', fontweight='bold')
    plt.ylabel('brainPAD_standardized', fontweight='bold')
    plt.title('Violin Plot', fontweight='bold')

    plt.savefig('III_7_40_plot.svg')


def generate_generic_data():
    np.random.seed(42)  # Set seed for reproducibility
    n = 100  # Number of samples per group
    groups = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

    data = {
        'Group': np.repeat(groups, n),
        'BrainPAD': np.concatenate([
            np.random.normal(0, 1, n),  # Group 1
            np.random.normal(1, 1, n),  # Group 2
            np.random.normal(2, 1, n),  # Group 3
            np.random.normal(3, 1, n)  # Group 4
        ])
    }

    return pd.DataFrame(data)


def rain_cloud_plot_V(df):

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Define colors for the groups
    grey_colors_light = '#B4BBBB'
    gray_colors_dark = '#858E8D'

    blue_colors_light = '#70B7CE'
    blue_colors_dark = '#348AA7'

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palettes for the groups
    palette_red_light = ['#CE9AA3', '#D07986', '#DC576B']
    palette_red_dark = ['#B67781', '#B65362', '#B52F43']

    palette_orange_light = ['#E2A98C', '#EC9468', '#F17D47']
    palette_orange_dark = ['#CC8E79', '#DA7949', '#DC5614']

    # Violin plot with no fill and switched axes (pliks18TH on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='pliks18TH',
        y='brainPAD_standardized',
        palette=[gray_colors_dark] + palette_red_dark,  # Color palette for violin outlines
        linewidth=2.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    palette_1 = [grey_colors_light] + palette_red_light

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(df['pliks18TH'].unique()):
        sns.boxplot(
            data=df[df['pliks18TH'] == group],
            x='pliks18TH',
            y='brainPAD_standardized',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='pliks18TH',
        y='brainPAD_standardized',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=palette_1,  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(df['pliks18TH'].unique()):
        group_mean = df[df['pliks18TH'] == group]['brainPAD_standardized'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('pliks18TH', fontweight='bold')
    plt.ylabel('brainPAD_standardized', fontweight='bold')
    plt.title('Violin Plot for brainPAD_standardized by grupo', fontweight='bold')

    plt.savefig('V_7_40_plot.svg')


def rain_cloud_plot_IV(df_prev_copy):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    df = df_prev_copy

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['grupo'].unique()

    for group in groups:
        x_values[group] = df[df['grupo'] == group]['Delta_BAG_standardized'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    grey_colors_light = '#B4BBBB'
    gray_colors_dark = '#858E8D'
    blue_colors_light = '#D8BFD8'
    blue_colors_dark = '#7E6A9C'
    red_colors_light = '#BF6673'
    red_colors_dark = '#C42840'
    orange_colors_light = '#A3B583'
    orange_colors_dark = '#556B2F'

    # Generate gradient colors for different groups
    palette_1 = interpolate_colors(hex_to_rgb(blue_colors_light), hex_to_rgb(orange_colors_light), len(groups))

    # Copy and modify the data
    df = df_prev_copy.copy()
    df['grupo'] = df['grupo'].replace('Crazy', 'PE_1-3')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with no fill and switched axes (Group on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='grupo',
        y='Delta_BAG_standardized',
        palette=[blue_colors_dark, orange_colors_dark],  # Color palette for violin outlines
        linewidth=3.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(groups):
        sns.boxplot(
            data=df[df['grupo'] == group],
            x='grupo',
            y='Delta_BAG_standardized',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='grupo',
        y='Delta_BAG_standardized',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=[blue_colors_light, orange_colors_light],  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(groups):
        group_mean = df[df['grupo'] == group]['Delta_BAG_standardized'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('grupo', fontweight='bold')
    plt.ylabel('Delta_BAG_standardized', fontweight='bold')
    plt.title('Violin Plot', fontweight='bold')

    plt.show()


def rain_cloud_plot_VI(df_prev_copy):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from collections import OrderedDict

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    df = df_prev_copy
    df['grupo_cat'] = df['grupo_cat'].replace('Enfermos', 'PEs')
    df['grupo_cat'] = df['grupo_cat'].replace('Controles', 'Controls')

    # Prepare the data for the raincloud plot
    x_values = OrderedDict()
    groups = df['grupo_cat'].unique()

    for group in groups:
        x_values[group] = df[df['grupo_cat'] == group]['DeltaBrainPAD'].values

    vals, names, xs = [], [], []
    for i, item in enumerate(x_values):
        vals.append(x_values[item])
        names.append(item)
        xs.append(np.random.normal(i, 0.03, x_values[item].shape[0]))

    # Define the colors and palette (can be customized)
    purple_colors_light = '#C59CDC'  # Light purple
    purple_colors_dark = '#6A3D9A'  # Dark purple
    green_colors_light = '#A6E6A1'  # Light green
    green_colors_dark = '#228B22'  # Dark green

    # Generate gradient colors for different groups
    palette_1 = interpolate_colors(hex_to_rgb(purple_colors_light), hex_to_rgb(green_colors_light), len(groups))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot with no fill and switched axes (Group on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='grupo_cat',
        y='DeltaBrainPAD',
        palette=[purple_colors_dark, green_colors_dark],  # Color palette for violin outlines
        linewidth=3.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(groups):
        sns.boxplot(
            data=df[df['grupo_cat'] == group],
            x='grupo_cat',
            y='DeltaBrainPAD',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='grupo_cat',
        y='DeltaBrainPAD',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=[purple_colors_light, green_colors_light],  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(groups):
        group_mean = df[df['grupo_cat'] == group]['DeltaBrainPAD'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('grupo_cat', fontweight='bold')
    plt.ylabel('DeltaBrainPAD', fontweight='bold')
    plt.title('Violin Plot', fontweight='bold')

    plt.show()


def rain_cloud_plot_VIII(df):

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Define colors for the groups
    purple_colors_light = '#C59CDC'  # Light purple
    purple_colors_dark = '#6A3D9A'  # Dark purple

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palettes for the groups
    palette_green_light = ['#A6E6A1', '#7AC96B', '#4CA743']  # Shades of green transitioning from light
    palette_green_dark = ['#67C477', '#3E9E31', '#228B22']  # Darker shades of green

    # Violin plot with no fill and switched axes (pliks18TH on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='Grupo_ordinal',
        y='DeltaBrainPAD',
        palette= [purple_colors_dark] + ['#67C477'] + ['#3E9E31', '#228B22'],  # Color palette for violin outlines
        linewidth=2.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    palette_1 =[purple_colors_light] + ['#A6E6A1'] + ['#7AC96B', '#4CA743']

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(sorted(df['Grupo_ordinal'].unique())):
        sns.boxplot(
            data=df[df['Grupo_ordinal'] == group],
            x='Grupo_ordinal',
            y='DeltaBrainPAD',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='Grupo_ordinal',
        y='DeltaBrainPAD',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=palette_1,  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(sorted(df['Grupo_ordinal'].unique())):
        group_mean = df[df['Grupo_ordinal'] == group]['DeltaBrainPAD'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('Grupo_ordinal', fontweight='bold')
    plt.ylabel('DeltaBrainPAD', fontweight='bold')
    plt.title('Violin Plot for brainPAD_standardized by grupo', fontweight='bold')

    plt.show()


def rain_cloud_plot_VII(df):

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Function to interpolate between two colors
    def interpolate_colors(start_color, end_color, n):
        start_color = np.array(start_color)
        end_color = np.array(end_color)
        colors = [start_color + (end_color - start_color) * t for t in np.linspace(0, 1, n)]
        hex_colors = [rgb_to_hex(color) for color in colors]
        return hex_colors

    # Define colors for the groups
    purple_colors_light = '#C59CDC'  # Light purple
    purple_colors_dark = '#6A3D9A'  # Dark purple

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palettes for the groups
    palette_green_light = ['#A6E6A1', '#7AC96B', '#4CA743']  # Shades of green transitioning from light
    palette_green_dark = ['#67C477', '#3E9E31', '#228B22']  # Darker shades of green

    # Violin plot with no fill and switched axes (pliks18TH on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='Grupo_ordinal',
        y='DeltaBrainPAD',
        palette= ['#67C477'] + [purple_colors_dark] + ['#3E9E31', '#228B22'],  # Color palette for violin outlines
        linewidth=2.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    palette_1 = ['#A6E6A1'] + [purple_colors_light] + ['#7AC96B', '#4CA743']

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(sorted(df['Grupo_ordinal'].unique())):
        sns.boxplot(
            data=df[df['Grupo_ordinal'] == group],
            x='Grupo_ordinal',
            y='DeltaBrainPAD',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    sns.stripplot(
        data=df,
        x='Grupo_ordinal',
        y='DeltaBrainPAD',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=palette_1,  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(sorted(df['Grupo_ordinal'].unique())):
        group_mean = df[df['Grupo_ordinal'] == group]['DeltaBrainPAD'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('Grupo_ordinal', fontweight='bold')
    plt.ylabel('DeltaBrainPAD', fontweight='bold')
    plt.title('Violin Plot for brainPAD_standardized by grupo', fontweight='bold')

    plt.show()


def rain_cloud_plot_colors(df):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Function to convert HEX to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Function to convert RGB to HEX
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))

    # Define colors for the groups
    grey_colors_light = '#B4BBBB'
    gray_colors_dark = '#858E8D'

    blue_colors_light = '#70B7CE'
    blue_colors_dark = '#348AA7'

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palettes for the groups
    palette_red_light = ['#CE9AA3', '#D07986', '#DC576B']
    palette_red_dark = ['#B67781', '#B65362', '#B52F43']

    palette_orange_light = ['#E2A98C', '#EC9468', '#F17D47']
    palette_orange_dark = ['#CC8E79', '#DA7949', '#DC5614']

    # Violin plot with no fill and switched axes (pliks18TH on x-axis, BrainPAD on y-axis)
    sns.violinplot(
        data=df,
        x='pliks18TH',
        y='brainPAD_standardized',
        palette=[gray_colors_dark] + palette_red_dark,  # Color palette for violin outlines
        linewidth=2.5,  # Line width for the violin outline
        fill=False,
        ax=ax,
        inner=None
    )

    palette_1 = [grey_colors_light] + palette_red_light

    # Overlay a boxplot with only the outline and no fill (facecolor='none')
    for i, group in enumerate(df['pliks18TH'].unique()):
        sns.boxplot(
            data=df[df['pliks18TH'] == group],
            x='pliks18TH',
            y='brainPAD_standardized',
            width=0.5,  # Narrower box width
            boxprops=dict(facecolor='none', edgecolor=palette_1[i], linewidth=2),  # Outline for each box with its own color
            whiskerprops=dict(linewidth=2, color=palette_1[i]),  # Thicker whiskers with corresponding color
            capprops=dict(linewidth=2, color=palette_1[i]),  # Thicker caps with corresponding color
            medianprops=dict(linewidth=2, color=palette_1[i]),  # Thicker median line with corresponding color
            showfliers=False,  # Optional: Hide outliers
            ax=ax
        )

    # Scatter plot to match the violin outline colors
    # Determine the last group and top 10 highest brainPAD_standardized
    last_group = df['pliks18TH'].unique()[-1]
    last_group_top_7 = df[df['pliks18TH'] == last_group].nlargest(7, 'brainPAD_standardized')

    sns.stripplot(
        data=df,
        x='pliks18TH',
        y='brainPAD_standardized',
        jitter=True,  # Add some jitter to the points
        size=7,  # Control point size
        palette=palette_1,  # Match scatter point color to the violin outline
        alpha=0.4,
        linewidth=0,  # Line width for scatter point edges
        ax=ax
    )

    for ID in last_group_top_7['ID'].values.tolist():
        print(ID)

    print(last_group_top_7)

    # Highlight top 10 IDs in blue for the last group
    ax.scatter(
        [last_group] * len(last_group_top_7),  # Position on the x-axis for the last group
        last_group_top_7['brainPAD_standardized'],  # Position on the y-axis
        color=blue_colors_dark,  # Blue color for the top 10 points
        s=70,  # Size of the points
        zorder=3, # Ensure it appears on top of other elements
        alpha=0.4
    )

    # Adding black points and labels for the mean for each group
    for i, group in enumerate(df['pliks18TH'].unique()):
        group_mean = df[df['pliks18TH'] == group]['brainPAD_standardized'].mean()

        # Add black point at the mean position
        ax.scatter(
            i,  # Position on the x-axis
            group_mean,  # Position on the y-axis (mean value)
            color='black',  # Black color for the point
            s=70,  # Size of the point
            zorder=3  # Ensure it appears on top of other elements
        )

        # Add text label for the mean with a box around it
        ax.text(
            i,  # Position on the x-axis
            group_mean + 0.1,  # Position on the y-axis (mean + small offset)
            f'Mean: {group_mean:.2f}',  # Text label showing the mean
            horizontalalignment='center',
            size='small',
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )

    # Customizing plot
    plt.xlabel('pliks18TH', fontweight='bold')
    plt.ylabel('brainPAD_standardized', fontweight='bold')
    plt.title('Violin Plot for brainPAD_standardized by group', fontweight='bold')

    plt.show()


def execute_in_val_and_test_NN_CV(data_train_filtered, edades_train, data_val_filtered, edades_val, lista, regresor, n_features, fold, params):

    # identifico en método de regresión
    regresor_used = lista[9]

    regresor.fit(data_train_filtered, edades_train, data_val_filtered, edades_val, fold, 639, params['layer_size'], lr=params['learning_rate'], weight_decay=1e-4, dropout=params['drop_out'], patience=params['epochs'], batch_size=params['batch_size'])



def plot_violin_2_grous(df: pd.DataFrame, p_text: str, group_col: str = 'Group', value_col: str = 'BrainPAD'):
    """
    Dibuja un violinplot de 2 grupos (p.ej., Controls vs COVID) para `value_col`.
    - p_text: cadena ya formateada para el p-valor (p.ej., format_small_number(p_val)).
    - Usa solo matplotlib y un único gráfico.
    """
    # Orden: Controls primero si existe
    groups = df[group_col].dropna().unique().tolist()
    if 'Controls' in groups and len(groups) == 2:
        order = ['Controls'] + [g for g in groups if g != 'Controls']
    else:
        order = sorted(groups)[:2]

    data = [df.loc[df[group_col] == g, value_col].dropna().values for g in order]

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    ax.set_xticks([1, 2]); ax.set_xticklabels(order)
    ax.set_xlabel(group_col)
    if value_col.lower().startswith('brainpad'):
        ax.set_ylabel('BrainPAD (años)')
        ax.set_title('BrainPAD por grupo')
    else:
        ax.set_ylabel(value_col)
        ax.set_title(f'{value_col} por grupo')

    if isinstance(p_text, str) and p_text:
        ax.text(0.98, 0.95, f'p = {p_text}', transform=ax.transAxes, ha='right', va='top')

    plt.tight_layout()
    return fig, ax

def plot_brain_age_vs_age(df: pd.DataFrame, age_col: str = 'Age', brain_age_col: str = 'BrainAge'):
    """
    Dispersión BrainAge vs Age con línea identidad (y=x).
    - Usa solo matplotlib y un único gráfico.
    """
    x = pd.to_numeric(df[age_col], errors='coerce')
    y = pd.to_numeric(df[brain_age_col], errors='coerce')
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = plt.gca()

    ax.scatter(x, y, s=18, alpha=0.8)
    mn = float(np.nanmin([x.min(), y.min()]))
    mx = float(np.nanmax([x.max(), y.max()]))
    ax.plot([mn, mx], [mn, mx], linewidth=1.0)

    ax.set_xlabel('Edad (años)')
    ax.set_ylabel('Brain Age (años)')
    ax.set_title('Brain Age vs Edad')
    plt.tight_layout()
    return fig, ax


import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

def pearson_safe(y, yhat):
    if len(y) < 2 or np.all(y==y[0]) or np.all(yhat==yhat[0]):
        return np.nan
    r, _ = pearsonr(y, yhat)
    return r

def r2_safe(y, yhat):
    if len(y) < 2:
        return np.nan
    try:
        return r2_score(y, yhat)
    except Exception:
        ss_res = np.sum((yhat - y)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

def bootstrap_ci(y, yhat, fn, B=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 2: return (np.nan, np.nan, np.nan)
    vals = np.empty(B, dtype=float)
    for b in range(B):
        ii = rng.integers(0, n, size=n)
        vals[b] = fn(y[ii], yhat[ii])
    return tuple(np.percentile(vals, [2.5,97.5]))

def coerce_sex(series):
    mapping = {1:"M",0:"F","1":"M","0":"F","M":"M","F":"F",
               "Male":"M","Female":"F","Hombre":"M","Mujer":"F"}
    return series.map(mapping)

def summarize_metrics(df, y_col, yhat_col, sex_col=None, B=5000, seed=42):
    df2 = df[[y_col, yhat_col] + ([sex_col] if sex_col else [])].replace([np.inf,-np.inf], np.nan).dropna()
    y  = df2[y_col ].to_numpy(float)
    yh = df2[yhat_col].to_numpy(float)

    def pack(y, yh, label):
        mae = mean_absolute_error(y, yh)
        r   = pearson_safe(y, yh)
        R2  = r2_safe(y, yh)
        mae_ci = bootstrap_ci(y, yh, lambda a,b: mean_absolute_error(a,b), B=B, seed=seed)
        r_ci   = bootstrap_ci(y, yh, pearson_safe, B=B, seed=seed)
        R2_ci  = bootstrap_ci(y, yh, r2_safe, B=B, seed=seed)
        return {
            "Grupo": label, "n": len(y),
            "MAE": mae, "MAE_CI_2.5%": mae_ci[0], "MAE_CI_97.5%": mae_ci[1],
            "r": r, "r_CI_2.5%": r_ci[0], "r_CI_97.5%": r_ci[1],
            "R²": R2, "R²_CI_2.5%": R2_ci[0], "R²_CI_97.5%": R2_ci[1],
        }

    rows = [pack(y, yh, "Total")]
    if sex_col:
        sex = coerce_sex(df2[sex_col])
        m, f = (sex=="M").to_numpy(), (sex=="F").to_numpy()
        if m.sum() >= 2: rows.append(pack(y[m], yh[m], "Varón (M)"))
        if f.sum() >= 2: rows.append(pack(y[f], yh[f], "Mujer (F)"))
    return pd.DataFrame(rows)