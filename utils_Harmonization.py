from sklearn.preprocessing import LabelEncoder
from neuroharmonize.neuroharmonize import harmonizationLearn
from neuroharmonize.neuroharmonize.harmonizationLearn import saveHarmonizationModel
from neuroharmonize.neuroharmonize.harmonizationApply import loadHarmonizationModel
from neuroharmonize.neuroharmonize import harmonizationApply


import pandas as pd
import numpy as np
import os

def clean_data(datos_todos):

    # filtro los datos que me interesan; No NKI-Rockland; No CoRR; Sólo morfo; solo edades entre 18 y 60
    datos_todos = datos_todos[datos_todos['Patologia'] != 'COVID']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'ControlNuevo']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_cefalea_Resto']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_cefalea_2as']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Control_seleccionado']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Control_Resto']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_EP_elegidos']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_CR_elegidos']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Depresion_NKI']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Depresion_CoRR']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'dolor_de_cabeza_repetido_o_severo']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_NKI']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Convulsions_seizures']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'epilepsia']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'TBI']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'defectos_nacimiento']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'intox_plomo']
    # datos_todos = datos_todos[datos_todos['Patologia'] != 'Problema_lenguaje']
    # datos_todos = datos_todos[datos_todos['Patologia'] != 'tics_vocales']
    # datos_todos = datos_todos[datos_todos['Patologia'] != 'tics_motores']
    # datos_todos = datos_todos[datos_todos['Patologia'] != 'dislexia']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Prob_aprendizaje']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Hiperactividad']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Prob_atención']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Autismo']
    # datos_todos = datos_todos[datos_todos['Patologia'] != 'Sonambulismo']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'moja_cama']
    # datos_todos = datos_todos[datos_todos['Patologia'] != 'prob_intensitnal']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Cancer']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Ataque_corazon']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'prob_coronario']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'prob_valvs']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Hipercolesterolemia']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Hipertension']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Hipotension']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Arritmia']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'ACV']
    # datos_todos = datos_todos[datos_todos['Patologia'] != 'IBS']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Crohn']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Colitis']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Reflujo']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Hepatitis']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'DB-1']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'DB-2']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Hiper_th']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Hipo_th']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'HIV']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Artritis']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Osteoporosis']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Enfisema']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Acne_sev']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Psoriasis']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Fatiga_Cronica']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Fibromialgia']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Bipolar']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'MCI']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Intento_Suicidio']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'OCD']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'ansiedad_social']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'PTSD']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'ataques_pánico']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'otros_ansiedad']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'ADHD']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Anorexia']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Bulimia']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Alzheimer']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Huntington']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Meningitis']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Esclerosis_multiple']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Parkinson']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Enfermedad_de_Lyme']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Dolor_MSK_A']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Dolor_MSK_B']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'SQZFR_otros']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'SQZFR_cron']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'SQZFR_primer_ep']
    datos_todos = datos_todos[datos_todos['Patologia'] != 'Bipolar']

    # Quito dos escáneres que tienen una n muy baja, y me joden la armonización
    datos_todos = datos_todos[datos_todos['Escaner'] != 'CoRR_LMU_3_Siemens_TrioTim']
    datos_todos = datos_todos[datos_todos['Escaner'] != 'CoRR_Utah_1_Siemens_TrioTim']

    # Estos escáneres joden la armonización en el rango 35-75 años
    # datos_todos = datos_todos[datos_todos['Escaner'] != 'CoRR_NYU_1_Siemens_Allegra']
    # datos_todos = datos_todos[datos_todos['Escaner'] != 'CoRR_NKI_1_Siemens_TrioTim']
    # datos_todos = datos_todos[datos_todos['Escaner'] != 'CoRR_IACAS_GE_Signa HDx']
    datos_todos = datos_todos[datos_todos['BD'] != 'NKI']

    return datos_todos

def harmonize_data_cleared(datos_todos):

    # uso la armonizacion de ComBatGAM para armonizar todos mis datos de entrenamiento

    lista_maquinas_todos = datos_todos['Escaner'].values
    etiv_todos = datos_todos['eTIV'].values
    edades_todos = datos_todos['Edad']
    bo_todos = datos_todos['Bo'].values
    sex_todos = datos_todos['sexo(M=1;F=0)'].values
    IDs = datos_todos['ID'].values
    BDs = datos_todos['DataBase'].values
    Patologia = datos_todos['Patologia'].values
    datos_todos = datos_todos.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad'], axis=1)

    features_names = datos_todos.columns.tolist()

    # armonizo las características con la edad como covariable usando ComBat
    # incluyo los escáneres y el sexo como LabelEncoder
    LE = LabelEncoder()
    datos_maquinas_num = pd.DataFrame(LE.fit_transform(lista_maquinas_todos))
    datos_sex_num = pd.DataFrame(sex_todos)
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    # Fit and transform the data

    # monto los datos de entrada y la matriz de covariables
    # Esta harmonización es la armonización de la base de datos de creación del modelo. No hace falta guardarla. No hace falta una referencia.
    datos_array = datos_todos.values
    d = {'SITE': datos_maquinas_num.values.tolist(), 'SEX': np.squeeze(datos_sex_num.values).tolist(),
          'ETIV': etiv_todos.tolist(), 'AGE': edades_todos.values.tolist()}
    covars = pd.DataFrame(data=d)
    # my_model, datos = harmonizationLearn(datos_array, covars, eb=True)

    # Si se quiere hacer la armonización con la parte de GAM. Añadir los términos de suavizado. Para tantas características como tenemos se alarga mucho el proceso
    my_model, datos = harmonizationLearn(datos_array, covars, smooth_terms=['AGE'], eb=False)

    # Estos son mis datos de entrenamiento armonizados
    datos = pd.DataFrame(datos, columns=features_names)
    datos['ID'] = IDs
    datos['DataBase'] = BDs
    datos['Edad'] = edades_todos.values
    datos['sexo(M=1;F=0)'] = sex_todos
    datos['Escaner'] = lista_maquinas_todos
    datos['Patologia'] = Patologia
    datos['Bo'] = bo_todos
    datos['eTIV'] = etiv_todos

    return datos

def harmonize_data_cleared_neuroHarmonize_pliks_categorical(
    datos_todos,
    ref_site=None,
    smooth_terms=None,
    smooth_term_bounds=(None, None),
    eb=True,
    return_s_data=False,
    seed=None):
    """
    Harmonize data using the rpomponio/neuroHarmonize library, treating 'pliks18TH'
    as a categorical variable (0,1,2,3) via dummy encoding. Optionally specify
    'smooth_terms' for non-linear (GAM) effects on e.g. 'AGE' or 'ETIV'.

    Parameters
    ----------
    datos_todos : pd.DataFrame
        Must contain:
          - morphological feature columns
          - 'Escaner' (scanner), 'Edad', 'sexo(M=1;F=0)', 'ID', 'DataBase',
            'Patologia', 'Bo', 'pliks18TH', 'eTIV'
    ref_site : str or int, optional
        Batch label to use as reference. If None, uses pooled approach.
    smooth_terms : list of str, optional
        Covariate names (in covars) to model as splines for ComBat-GAM.
    smooth_term_bounds : tuple of float, optional
        Bounds for smoothing if only one smooth term. (None, None) by default.
    eb : bool, default True
        Use Empirical Bayes.
    return_s_data : bool, default False
        If True, also returns standardized data.
    seed : int, optional
        Random seed for reproducibility in smoothing.

    Returns
    -------
    harmonized_df : pd.DataFrame
        DataFrame with morphological columns harmonized and original metadata reattached.
    s_data : np.ndarray, optional
        If return_s_data=True, the standardized residuals are returned, shape [N_samples, N_features].
    """

    if smooth_terms is None:
        smooth_terms = []

    # 1) Extract relevant metadata columns
    lista_maquinas_todos = datos_todos['Escaner'].values
    etiv_todos = datos_todos['eTIV'].values
    edades_todos = datos_todos['Edad'].values
    bo_todos = datos_todos['Bo'].values
    sex_todos = datos_todos['sexo(M=1;F=0)'].values
    IDs = datos_todos['ID'].values
    BDs = datos_todos['DataBase'].values
    Patologia = datos_todos['Patologia'].values
    pliks = datos_todos['pliks18TH'].values
    # e.g. pliks might be [0,1,2,3], representing categories

    # 2) Drop metadata columns, leaving only morphological features
    drop_cols = [
        'ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase',
        'Patologia', 'Edad', 'pliks18TH', 'eTIV'
    ]
    datos_features = datos_todos.drop(columns=drop_cols)
    feature_names = datos_features.columns.tolist()

    # 3) Convert scanner to numeric "SITE" via LabelEncoder
    LE = LabelEncoder()
    site_numeric = LE.fit_transform(lista_maquinas_todos)  # e.g. [0,1,2] etc.

    # 4) Create a DataFrame for covariates
    # We'll store 'AGE', 'SEX', 'ETIV' as numeric columns.
    # Then we convert pliks to one-hot columns (for categories 0,1,2,3).
    covars_base = pd.DataFrame({
        'SITE': site_numeric,
        'SEX': sex_todos.astype(float),
        'AGE': edades_todos.astype(float),
        'ETIV': etiv_todos.astype(float),
    })

    # 5) Make 'pliks18TH' categorical via dummy columns, e.g. pliks_0, pliks_1, pliks_2, pliks_3
    # By default, pd.get_dummies won't drop any category so we get one column per pliks value.
    df_pliks_dummy = pd.get_dummies(pliks.astype(int), prefix='pliks', dtype=float)
    # Concatenate
    covars = pd.concat([covars_base, df_pliks_dummy], axis=1)

    # 6) Prepare the morphological data array for ComBat
    # neuroHarmonize expects shape [N_samples, N_features]
    data_array = datos_features.values  # shape [N_samples, N_features]

    # 7) Call harmonizationLearn from rpomponio/neuroHarmonize
    results = harmonizationLearn(
        data=data_array,
        covars=covars,
        eb=eb,
        smooth_terms=smooth_terms,              # e.g. ['AGE'] if you want a spline on age
        smooth_term_bounds=smooth_term_bounds,
        ref_batch=ref_site,                    # integer or string matching 'SITE' column
        return_s_data=return_s_data,
        seed=seed
    )

    # 8) Unpack results
    if return_s_data:
        model, bayes_data, s_data = results
    else:
        model, bayes_data = results

    # bayes_data is [N_samples, N_features], same shape as data_array
    # 9) Convert bayes_data back into a DataFrame, reattach metadata
    harmonized_df = pd.DataFrame(bayes_data, columns=feature_names)

    # Reattach original metadata columns
    harmonized_df['ID'] = IDs
    harmonized_df['DataBase'] = BDs
    harmonized_df['Edad'] = edades_todos
    harmonized_df['sexo(M=1;F=0)'] = sex_todos
    harmonized_df['Escaner'] = lista_maquinas_todos
    harmonized_df['Patologia'] = Patologia
    harmonized_df['Bo'] = bo_todos
    harmonized_df['eTIV'] = etiv_todos
    harmonized_df['pliks18TH'] = pliks  # original pliks values

    # 10) Return harmonized DataFrame, plus s_data if requested
    if return_s_data:
        return harmonized_df, s_data
    else:
        return harmonized_df


def learn_harmonization(Datos_Ref, datos_source, harmo_name):

    # Aprendo una armonización para los datos de entrenamiento y los controles que luego aplico sobre los casos de migraña
    # Junto los datos
    pliks_flag = False

    Datos_Ref['Escaner'] = 'zarmonization_1' # Zarmoniation porque coge el ref batch por orden alfabético

    if 'pliks18TH' in datos_source.columns.tolist():
        pliks_flag = True
        pliks = datos_source['pliks18TH'].values
        datos_source = datos_source.drop(['pliks18TH', 'pliks20TH'], axis='columns')
        datos_source['Escaner'] = 'CARDIFF_ESCANER'
        datos_source = datos_source.rename(columns={'sexo': 'sexo(M=1;F=0)'})
        datos_source['Bo'] = '3.0T'
        datos_source = datos_source[Datos_Ref.columns.tolist()]
    datos_source = datos_source.loc[:, ~datos_source.columns.duplicated()]
    datos_source = datos_source[Datos_Ref.columns.tolist()]
    datos_todos = pd.concat([datos_source, Datos_Ref])

    # Limpio los elementos que no son features per se para la armonización
    lista_maquinas_todos = datos_todos['Escaner'].values
    etiv_todos = datos_todos['eTIV'].values
    edades_todos = datos_todos['Edad']
    bo_todos = datos_todos['Bo'].values
    sex_todos = datos_todos['sexo(M=1;F=0)'].values
    IDs = datos_todos['ID'].values
    BDs = datos_todos['DataBase'].values
    Patologia = datos_todos['Patologia'].values

    datos_todos = datos_todos.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad', 'eTIV'], axis=1)

    # guardo los nombres de las varibles
    names = datos_todos.columns.tolist()
    print(names)

    # incluyo los escáneres y el sexo como LabelEncoder
    LE = LabelEncoder()
    datos_maquinas_num = pd.DataFrame(LE.fit_transform(lista_maquinas_todos))
    LE_sex = LabelEncoder()
    datos_sex_num = pd.DataFrame(sex_todos)

    # Preparo los datos para la armonizacion 2
    datos_array = datos_todos.values
    d = {'SITE': datos_maquinas_num.values.ravel().tolist(), 'SEX': np.squeeze(datos_sex_num.values).tolist(),
         'ETIV': etiv_todos.tolist(), 'AGE': edades_todos.values.tolist()}
    covars = pd.DataFrame(data=d)

    # Aprendo la armonización, la aplico sobre los controles del escaner de la uva y la guardo
    # my_model, datos = harmonizationLearn(datos_array, covars, ref_batch=1)

    # Si quieres incluir la parte de GAM añadir los términos de suavizado. No vale la pena el tiempo de cómputo es muy largo
    # my_model, datos = harmonizationLearn(datos_array, covars, smooth_terms=['ETIV', 'AGE', 'SEX'], ref_batch=1, eb=False)
    my_model, datos = harmonizationLearn(datos_array, covars, ref_batch=1, smooth_terms=['AGE'], eb=False)

    if not os.path.isfile('/home/rafa/workRafa/ComBatGAM/'+harmo_name):
        saveHarmonizationModel(my_model, harmo_name)
        print('Modelo guardado')
    else:
        print('El modelo_noMCQR ya existe. Modelo no guardado.')

    # Estos son mis datos de entrenamiento armonizados
    datos = pd.DataFrame(datos, columns=names)
    datos['Edad'] = edades_todos.values
    datos['Escaner'] = lista_maquinas_todos
    datos['Bo'] = bo_todos
    datos['sexo(M=1;F=0)'] = sex_todos
    datos['eTIV'] = etiv_todos
    datos['ID'] = IDs
    datos['DataBase'] = BDs
    datos['Patologia'] = Patologia

    if pliks_flag:
        datos = datos[datos['Escaner'] != 'zarmonization_1']
        datos['pliks18TH'] = pliks

    # save_dir = config_parser.get("RESULTADOS", "Resultado_Modelo_BrainAge")
    # datos.to_csv(os.path.join(save_dir, 'datos_todos_armo1.csv'))

    return datos, my_model

def _prep_covars(df, site_map):
    # accept either 'sexo(M=1;F=0)' or 'sex'
    sex_col = 'sexo(M=1;F=0)' if 'sexo(M=1;F=0)' in df.columns else 'sex'
    covars = pd.DataFrame({
        'SITE': df['Escaner'].astype(str).map(site_map).fillna(-1).astype(int),
        'SEX' : df[sex_col].astype(int),
        'ETIV': df['eTIV'].astype(float),
        'AGE' : df['Edad'].astype(float),
    })
    # guard: unseen sites -> fail early (your apply() sets NaN, but better to catch)
    if (covars['SITE'] < 0).any():
        missing = df.loc[covars['SITE'] < 0, 'Escaner'].astype(str).unique()
        raise ValueError(f"Unseen SITE(s) in new data (not present in TRAIN): {missing}")
    return covars


def learn_harmonization_noRef(datos_todos, harmo_name):

    # Aprendo una armonización para los datos de entrenamiento y los controles que luego aplico sobre los casos de migraña

    # Limpio los elementos que no son features per se para la armonización
    lista_maquinas_todos = datos_todos['Escaner'].values
    etiv_todos = datos_todos['eTIV'].values
    edades_todos = datos_todos['Edad']
    bo_todos = datos_todos['Bo'].values
    sex_todos = datos_todos['sexo(M=1;F=0)'].values
    IDs = datos_todos['ID'].values
    BDs = datos_todos['DataBase'].values
    Patologia = datos_todos['Patologia'].values

    datos_todos = datos_todos.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad', 'eTIV'], axis=1)

    # guardo los nombres de las varibles
    names = datos_todos.columns.tolist()

    print(names)

    # incluyo los escáneres y el sexo como LabelEncoder
    LE = LabelEncoder()
    datos_maquinas_num = pd.DataFrame(LE.fit_transform(lista_maquinas_todos))
    LE_sex = LabelEncoder()
    datos_sex_num = pd.DataFrame(sex_todos)

    # Preparo los datos para la armonizacion 2
    datos_array = datos_todos.values
    d = {'SITE': datos_maquinas_num.values.ravel().tolist(), 'SEX': np.squeeze(datos_sex_num.values).tolist(),
         'ETIV': etiv_todos.tolist(), 'AGE': edades_todos.values.tolist()}
    covars = pd.DataFrame(data=d)

    # Aprendo la armonización, la aplico sobre los controles del escaner de la uva y la guardo
    # my_model, datos = harmonizationLearn(datos_array, covars, ref_batch=1)

    # Si quieres incluir la parte de GAM añadir los términos de suavizado. No vale la pena el tiempo de cómputo es muy largo
    my_model, datos = harmonizationLearn(datos_array, covars, smooth_terms=['AGE'], eb=True)

    if not os.path.isfile('/home/rafa/workRafa/ComBatGAM/'+harmo_name):
        saveHarmonizationModel(my_model, harmo_name)
        print('Modelo guardado')
    else:
        print('El modelo ya existe. Modelo no guardado.')

    # Estos son mis datos de entrenamiento armonizados
    datos = pd.DataFrame(datos, columns=names)
    datos['Edad'] = edades_todos.values
    datos['Escaner'] = lista_maquinas_todos
    datos['Bo'] = bo_todos
    datos['sexo(M=1;F=0)'] = sex_todos
    datos['eTIV'] = etiv_todos
    datos['ID'] = IDs
    datos['DataBase'] = BDs
    datos['Patologia'] = Patologia

    return datos, my_model

def _prep_covars(df, site_map):
    # accept either 'sexo(M=1;F=0)' or 'sex'
    sex_col = 'sexo(M=1;F=0)' if 'sexo(M=1;F=0)' in df.columns else 'sex'
    covars = pd.DataFrame({
        'SITE': df['Escaner'].astype(str).map(site_map).fillna(-1).astype(int),
        'SEX' : df[sex_col].astype(int),
        'ETIV': df['eTIV'].astype(float),
        'AGE' : df['Edad'].astype(float),
    })
    # guard: unseen sites -> fail early (your apply() sets NaN, but better to catch)
    if (covars['SITE'] < 0).any():
        missing = df.loc[covars['SITE'] < 0, 'Escaner'].astype(str).unique()
        raise ValueError(f"Unseen SITE(s) in new data (not present in TRAIN): {missing}")
    return covars


def apply_harmonization(datos_to_apply, datos_ref, ref_level, save_dir, name):

    if 'pliks18TH' in datos_to_apply.columns.tolist():
        pliks_flag = True
        pliks = datos_to_apply['pliks18TH'].values
        datos_to_apply = datos_to_apply.drop(['pliks18TH', 'pliks20TH'], axis='columns')
        datos_to_apply['Bo'] = '3.0T'
        datos_to_apply['Escaner'] = 'CARDIFF_ESCANER'
        datos_to_apply = datos_to_apply.rename(columns={'sexo': 'sexo(M=1;F=0)'})

    datos_to_apply = datos_to_apply[datos_ref.columns.tolist()]

    # Limpio los elementos que no son features per se para la armonización
    lista_maquinas_todos = datos_to_apply['Escaner'].values
    etiv_todos = datos_to_apply['eTIV'].values
    edades_todos = datos_to_apply['Edad']
    bo_todos = datos_to_apply['Bo'].values
    sex_todos = datos_to_apply['sexo(M=1;F=0)'].values
    IDs = datos_to_apply['ID'].values
    BDs = datos_to_apply['DataBase'].values
    Patologia = datos_to_apply['Patologia'].values
    datos_to_apply = datos_to_apply.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad', 'eTIV'], axis=1)

    # guardo los nombres de las varibles
    names = datos_to_apply.columns.tolist()

    # incluyo los escáneres y el sexo como LabelEncoder
    LE = LabelEncoder()
    datos_maquinas_num = pd.DataFrame(LE.fit_transform(lista_maquinas_todos))
    LE_sex = LabelEncoder()
    datos_sex_num = pd.DataFrame(LE_sex.fit_transform(sex_todos))

    # Preparo los datos para la armonizacion 2
    datos_array = datos_to_apply.values
    d = {'SITE': datos_maquinas_num.values.ravel().tolist(), 'SEX': np.squeeze(datos_sex_num.values).tolist(),
         'ETIV': etiv_todos.tolist(), 'AGE': edades_todos.values.tolist()}
    covars = pd.DataFrame(data=d)

    # 3) consistent SITE coding (replicates LabelEncoder(sorted) on TRAIN sites)
    train_sites = np.sort(datos_ref['Escaner'].astype(str).unique())
    site_map = {s: i for i, s in enumerate(train_sites)}

    # 4) covariates & data matrix
    datos_array = datos_to_apply.copy()

    # Cargo y aplico la armonización
    my_model = loadHarmonizationModel(save_dir+'/'+name)
    datos_harmonized_array = harmonizationApply(datos_array, covars, my_model, ref_level)

    # Estos son mis datos de entrenamiento armonizados
    datos = pd.DataFrame(datos_harmonized_array, columns=names)
    datos['ID'] = IDs
    datos['DataBase'] = BDs
    datos['Edad'] = edades_todos.values
    datos['sexo(M=1;F=0)'] = sex_todos
    datos['Escaner'] = lista_maquinas_todos
    datos['Patologia'] = Patologia
    datos['Bo'] = bo_todos
    datos['eTIV'] = etiv_todos

    return datos


def harmonize_data_cleared_pliks(datos_todos):

    # uso la armonizacion de ComBatGAM para armonizar todos mis datos de entrenamiento

    lista_maquinas_todos = datos_todos['Escaner'].values
    etiv_todos = datos_todos['eTIV'].values
    edades_todos = datos_todos['Edad']
    bo_todos = datos_todos['Bo'].values
    sex_todos = datos_todos['sexo(M=1;F=0)'].values
    IDs = datos_todos['ID'].values
    BDs = datos_todos['DataBase'].values
    Patologia = datos_todos['Patologia'].values
    pliks = datos_todos['pliks18TH'].values
    time = datos_todos['Time'].values
    datos_todos = datos_todos.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad', 'pliks18TH', 'Time'], axis=1)

    features_names = datos_todos.columns.tolist()

    # armonizo las características con la edad como covariable usando ComBat
    # incluyo los escáneres y el sexo como LabelEncoder
    LE = LabelEncoder()
    datos_maquinas_num = pd.DataFrame(LE.fit_transform(lista_maquinas_todos))
    datos_sex_num = pd.DataFrame(sex_todos)
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    # Fit and transform the data

    # monto los datos de entrada y la matriz de covariables
    # Esta harmonización es la armonización de la base de datos de creación del modelo. No hace falta guardarla. No hace falta una referencia.
    datos_array = datos_todos.values
    d = {'SITE': datos_maquinas_num.values.tolist(), 'SEX': np.squeeze(datos_sex_num.values).tolist(),
          'ETIV': etiv_todos.tolist(), 'AGE': edades_todos.values.tolist(), 'PLIKS': pliks.tolist()}
    covars = pd.DataFrame(data=d)
    # my_model, datos = harmonizationLearn(datos_array, covars, eb=True)

    # Si se quiere hacer la armonización con la parte de GAM. Añadir los términos de suavizado. Para tantas características como tenemos se alarga mucho el proceso
    my_model, datos = harmonizationLearn(datos_array, covars, smooth_terms=['ETIV', 'AGE', 'SEX', 'PLIKS'], eb=True)

    # Estos son mis datos de entrenamiento armonizados
    datos = pd.DataFrame(datos, columns=features_names)
    datos['ID'] = IDs
    datos['DataBase'] = BDs
    datos['Edad'] = edades_todos.values
    datos['sexo(M=1;F=0)'] = sex_todos
    datos['Escaner'] = lista_maquinas_todos
    datos['Patologia'] = Patologia
    datos['Bo'] = bo_todos
    datos['eTIV'] = etiv_todos
    datos['pliks18TH'] = pliks
    datos['Time'] = time

    return datos


def harmonize_eTIV(datos_todos):

    # uso la armonizacion de ComBatGAM para armonizar todos mis datos de entrenamiento

    lista_maquinas_todos = datos_todos['Escaner'].values
    etiv_todos = datos_todos['eTIV'].values
    edades_todos = datos_todos['Edad']
    bo_todos = datos_todos['Bo'].values
    sex_todos = datos_todos['sexo(M=1;F=0)'].values
    IDs = datos_todos['ID'].values
    BDs = datos_todos['DataBase'].values
    Patologia = datos_todos['Patologia'].values
    pliks = datos_todos['pliks18TH'].values
    time = datos_todos['Time'].values
    datos_todos = datos_todos.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad', 'pliks18TH', 'Time'], axis=1)

    features_names = datos_todos.columns.tolist()

    # armonizo las características con la edad como covariable usando ComBat
    # incluyo los escáneres y el sexo como LabelEncoder
    LE = LabelEncoder()
    datos_maquinas_num = pd.DataFrame(LE.fit_transform(lista_maquinas_todos))
    datos_sex_num = pd.DataFrame(sex_todos)
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    # Fit and transform the data

    # monto los datos de entrada y la matriz de covariables
    # Esta harmonización es la armonización de la base de datos de creación del modelo. No hace falta guardarla. No hace falta una referencia.
    datos_array = datos_todos.values
    d = {'SITE': datos_maquinas_num.values.tolist(), 'SEX': np.squeeze(datos_sex_num.values).tolist(),
         'AGE': edades_todos.values.tolist(), 'PLIKS': pliks.tolist()}
    covars = pd.DataFrame(data=d)
    # my_model, datos = harmonizationLearn(datos_array, covars, eb=True)

    # Si se quiere hacer la armonización con la parte de GAM. Añadir los términos de suavizado. Para tantas características como tenemos se alarga mucho el proceso
    my_model, datos = harmonizationLearn(datos_array, covars, smooth_terms=['AGE', 'SEX', 'PLIKS'], eb=True)

    # Estos son mis datos de entrenamiento armonizados
    datos = pd.DataFrame(datos, columns=features_names)
    datos['ID'] = IDs
    datos['DataBase'] = BDs
    datos['Edad'] = edades_todos.values
    datos['sexo(M=1;F=0)'] = sex_todos
    datos['Escaner'] = lista_maquinas_todos
    datos['Patologia'] = Patologia
    datos['Bo'] = bo_todos
    datos['eTIV_old'] = etiv_todos
    datos['pliks18TH'] = bo_todos
    datos['Time'] = time

    return datos

