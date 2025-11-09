from neuroharmonize.neuroharmonize import harmonizationLearn
from neuroharmonize.neuroharmonize.harmonizationLearn import saveHarmonizationModel
from neuroharmonize.neuroharmonize.harmonizationApply import loadHarmonizationModel
from neuroharmonize.neuroharmonize import harmonizationApply

from sklearn.preprocessing import LabelEncoder

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

def learn_harmonization(Datos_Ref, datos_source, harmo_name):

    # Aprendo una armonización para los datos de entrenamiento y los controles que luego aplico sobre los casos de migraña
    # Junto los datos

    Datos_Ref['Escaner'] = 'zarmonization_1' # Zarmoniation porque coge el ref batch por orden alfabético

    if 'pliks' in datos_source.columns.tolist()[4]:
        datos_source = datos_source.drop(['pliks18TH', 'pliks20TH'], axis='columns')
        datos_source['Escaner'] = 'CARDIFF_ESCANER'
        datos_source = datos_source.rename(columns={'sexo': 'sexo(M=1;F=0)'})
        datos_source['Bo'] = '3.0T'
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
    datos_todos = datos_todos.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad'], axis=1)

    # guardo los nombres de las varibles
    names = datos_todos.columns.tolist()

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
    my_model, datos = harmonizationLearn(datos_array, covars, ref_batch=1)

    # Si quieres incluir la parte de GAM añadir los términos de suavizado. No vale la pena el tiempo de cómputo es muy largo
    # my_model, datos = harmonizationLearn(datos_array, covars, smooth_terms=['ETIV', 'AGE', 'SEX'], ref_batch=1)

    if not os.path.isfile('/home/rafa/PycharmProjects/JoinData_FastSurfer_V2/'+harmo_name):
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

    # save_dir = config_parser.get("RESULTADOS", "Resultado_Modelo_BrainAge")
    # datos.to_csv(os.path.join(save_dir, 'datos_todos_armo1.csv'))

    return datos, my_model

def apply_harmonization(datos_to_apply, datos_ref, ref_level, save_dir, name):

    if 'pliks' in datos_to_apply.columns.tolist()[4]:
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
    datos_to_apply = datos_to_apply.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Patologia', 'Edad'], axis=1)

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

    # monto los datos de entrada y la matriz de covariables
    # Esta harmonización es la armonización de la base de datos de creación del modelo. No hace falta guardarla. No hace falta una referencia.
    datos_array = datos_todos.values
    d = {'SITE': datos_maquinas_num.values.tolist(), 'SEX': np.squeeze(datos_sex_num.values).tolist(),
         'ETIV': etiv_todos.tolist(), 'AGE': edades_todos.values.tolist()}
    covars = pd.DataFrame(data=d)
    my_model, datos = harmonizationLearn(datos_array, covars)

    # Si se quiere hacer la armonización con la parte de GAM. Añadir los términos de suavizado. Para tantas características como tenemos se alarga mucho el proceso
    # my_model, datos = harmonizationLearn(datos_array, covars, smooth_terms=['ETIV', 'AGE', 'SEX'])

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

