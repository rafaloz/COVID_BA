from sklearn.preprocessing import LabelEncoder
from neuroharmonize.neuroharmonize import harmonizationLearn
from neuroharmonize.neuroharmonize.harmonizationLearn import saveHarmonizationModel
from neuroharmonize.neuroharmonize.harmonizationApply import loadHarmonizationModel
from neuroharmonize.neuroharmonize import harmonizationApply


import pandas as pd
import numpy as np
import os

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

