import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib
import numpy as np
import altair as alt
from collections import Counter
from datetime import datetime, timedelta

#Cargar los files a la carpeta: 
# model_rf_CPC_v3, model_xgboost_CPC_v3, model_NN_CPC_v3, scaler_model_CPC_v3, pca_model_CPC_v3, scaler_NN_model_CPC_v3
# model_rf_CPM_v3, model_xgboost_CPM_v3, model_NN_CPM_v3, scaler_model_CPM_v3, pca_model_CPM_v3, scaler_NN_model_CPMC_v3
# model_rf_CTR_v3, model_xgboost_CTR_v3, model_NN_CTR_v3, scaler_model_CTR_v3, pca_model_CTR_v3, scaler_NN_model_CTR_v3
# model_rf_CPV_v3, model_xgboost_CPV_v3, model_NN_CPV_v3, scaler_model_CPV_v3, pca_model_CPV_v3, scaler_NN_model_CPV_v3
# search_bench.csv, fb_bench.csv, bench_yt.csv, df_histo.csv
# requirements.txt


st.title('Uso del modelo')

version = '5'

# Cargar los modelos desde los archivos
RF_CPC =  joblib.load("model_rf_CPC_v"+version+".joblib")
xgboost_CPC = xgb.XGBRegressor()
xgboost_CPC.load_model("model_xgboost_CPC_v"+version+".json")
NN_CPC = joblib.load("model_NN_CPC_v"+version+".joblib")

RF_CPC_sinClient =  joblib.load("model_rf_CPC_v"+version+"_sinClient.joblib")
xgboost_CPC_sinClient = xgb.XGBRegressor()
xgboost_CPC_sinClient.load_model("model_xgboost_CPC_v"+version+"_sinClient.json")
NN_CPC_sinClient = joblib.load("model_NN_CPC_v"+version+"_sinClient.joblib")
loaded_scaler_NN_CPC_sinClient = joblib.load('scaler_NN_model_CPC_v'+version+'_sinClient.joblib')

loaded_scaler_CPC = joblib.load('scaler_model_CPC_v'+version+'.joblib')
loaded_pca_CPC = joblib.load('pca_model_CPC_v'+version+'.joblib')
loaded_scaler_NN_CPC = joblib.load('scaler_NN_model_CPC_v'+version+'.joblib')




RF_CPM =  joblib.load("model_rf_CPM_v"+version+".joblib")
xgboost_CPM = xgb.XGBRegressor()
xgboost_CPM.load_model("model_xgboost_CPM_v"+version+".json")
NN_CPM = joblib.load("model_NN_CPM_v"+version+".joblib")

RF_CPM_sinClient =  joblib.load("model_rf_CPM_v"+version+"_sinClient.joblib")
xgboost_CPM_sinClient = xgb.XGBRegressor()
xgboost_CPM_sinClient.load_model("model_xgboost_CPM_v"+version+"_sinClient.json")
NN_CPM_sinClient = joblib.load("model_NN_CPM_v"+version+"_sinClient.joblib")
loaded_scaler_NN_CPM_sinClient = joblib.load('scaler_NN_model_CPM_v'+version+'_sinClient.joblib')

loaded_scaler_CPM = joblib.load('scaler_model_CPM_v'+version+'.joblib')
loaded_pca_CPM = joblib.load('pca_model_CPM_v'+version+'.joblib')
loaded_scaler_NN_CPM = joblib.load('scaler_NN_model_CPM_v'+version+'.joblib')




RF_CTR =  joblib.load("model_rf_CTR_v"+version+".joblib")
xgboost_CTR = xgb.XGBRegressor()
xgboost_CTR.load_model("model_xgboost_CTR_v"+version+".json")
NN_CTR = joblib.load("model_NN_CTR_v"+version+".joblib")

RF_CTR_sinClient =  joblib.load("model_rf_CTR_v"+version+"_sinClient.joblib")
xgboost_CTR_sinClient = xgb.XGBRegressor()
xgboost_CTR_sinClient.load_model("model_xgboost_CTR_v"+version+"_sinClient.json")
NN_CTR_sinClient = joblib.load("model_NN_CTR_v"+version+"_sinClient.joblib")
loaded_scaler_NN_CTR_sinClient = joblib.load('scaler_NN_model_CTR_v'+version+'_sinClient.joblib')

loaded_scaler_CTR = joblib.load('scaler_model_CTR_v'+version+'.joblib')
loaded_pca_CTR = joblib.load('pca_model_CTR_v'+version+'.joblib')
loaded_scaler_NN_CTR = joblib.load('scaler_NN_model_CTR_v'+version+'.joblib')





RF_CPV =  joblib.load("model_rf_CPV_v"+version+".joblib")
xgboost_CPV = xgb.XGBRegressor()
xgboost_CPV.load_model("model_xgboost_CPV_v"+version+".json")
NN_CPV = joblib.load("model_NN_CPV_v"+version+".joblib")
loaded_scaler_CPV = joblib.load('scaler_model_CPV_v'+version+'.joblib')
loaded_pca_CPV = joblib.load('pca_model_CPV_v'+version+'.joblib')
loaded_scaler_NN_CPV = joblib.load('scaler_NN_model_CPV_v'+version+'.joblib')

def load_data(df_in):
    df = pd.read_csv(df_in+'.csv')
    df = df.drop("Unnamed: 0", axis=1)
    return df

# Cargar los datos
df = load_data('df_histo')

df_cpv = df.copy()
df_cpv = df_cpv[df_cpv['CPV']>0]
df_cpv = df_cpv[df_cpv['CPV']<2]


def load_clients(df_in):
    df = pd.read_csv(df_in+'.csv')
    return df

#df_clients = load_clients('Clients')
df_search_bench = load_clients('search_bench_v2')
df_search_bench = df_search_bench.set_index('Tipo Search')

df_FB_bench = load_clients('fb_bench')
df_FB_bench = df_FB_bench.set_index('Tipo FB')

df_YT_bench = load_clients('bench_yt')
df_YT_bench = df_YT_bench.set_index('Tipo YT')

df_dicc_industrias = load_clients('diccionario_industrias')
df_dicc_industrias = df_dicc_industrias.set_index('Search')


variables_modelo = xgboost_CPC.feature_names_in_

all_features = ['Año','Mes', 'Objective', 'Cost', 'Country', 'Media_type', 'Traffic_source','Format_New','Platform','Strategy','Plataforma'
                ,'Campaign_Type','Ecommerce','Service_Product','Semanas_Antiguedad','Client','Bench Gral CPC','Bench Search CPC','Bench GralSch CPL', 'Bench Search CPL',
                'Bench GralSch CTR', 'Bench Search CTR', 'Bench GralSch CR','Bench Search AvgCR','Bench GralFB CPC', 'Bench FB CPC','Bench GralFB CPAction', 'Bench FB CPAction',
                'Bench GralFB CTR', 'Bench FB CTR', 'Bench GralFB CR', 'Bench FB AvgCR','Bench GralYT CPV', 'Bench YT CPV', 'Bench GralYT CTR', 'Bench YT CTR','Bench GralYT VR', 'Bench FB AvgVR']

categorical_features = ['Objective', 'Country', 'Media_type', 'Traffic_source', 'Format_New','Platform','Strategy','Plataforma','Campaign_Type','Ecommerce','Service_Product','Client']



with st.sidebar:
    
    
    #Año = st.number_input('Año de la campaña a planificar',value=2023)
    #Mes = st.number_input('Mes de la campaña a planificar',value=11, max_value=12)

    def obtener_mes_mayoria(fecha_inicio_dt, fecha_fin_dt):
    
        def valor_mas_frecuente(lista):
            # Contar la frecuencia de cada elemento manualmente
            frecuencia = {}
            for elemento in lista:
                if elemento in frecuencia:
                    frecuencia[elemento] += 1
                else:
                    frecuencia[elemento] = 1
        
            # Encontrar el elemento con la frecuencia más alta
            valor_mayoria = max(frecuencia, key=frecuencia.get)
        
            return valor_mayoria
        
        # Crear un diccionario para almacenar la frecuencia de cada mes
        frecuencia_meses = []
        frecuencia_años = []
    
        # Iterar sobre cada día en el rango de fechas
        current_date = fecha_inicio_dt
    
    
        while current_date <= fecha_fin_dt:
            # Obtener el mes actual y actualizar el diccionario de frecuencia
            mes_actual = current_date.month
            año_actual = current_date.year
            frecuencia_meses.append(mes_actual)
            frecuencia_años.append(año_actual)
    
            # Moverse al siguiente día
            current_date += timedelta(days=1)
        mes_mayoria = valor_mas_frecuente(frecuencia_meses)
        año_mayoria = valor_mas_frecuente(frecuencia_años)
        # Encontrar el mes con la mayoría de los días
       
        return mes_mayoria, año_mayoria
    
    today = datetime.now()
    dias_campaña = 1
    d = st.date_input(    "Seleccionar el período de la campaña [DD.MM.AAAA]",(today,today),   format="DD.MM.YYYY")

    try:
        Año = today.year
        Mes = today.month
        fecha_inicio = d[0]
        fecha_fin = d[1]        
        dias_campaña = (fecha_fin-fecha_inicio).days + 1
        Mes, Año = obtener_mes_mayoria(fecha_inicio, fecha_fin)
    except:
        print('hubo algún error')
        pass



    
    Cost_periodo = st.number_input('Costo total del período de '+str(dias_campaña)+' días',value=1000)
    Objective = st.selectbox(    'Objetivo',    (['Purchase','Fans','Reach', 'Traffic', 'Category', 'Awareness','Product', 'Consideration',
                                                  'Conversion', 'Views','Landing Page Views', 'NoObjective', 'Discovery', 'Impressions','Clicks', 'Conversions', 'Whatsapp']))
    Country = st.selectbox(    'Country',    (['USA','Mexico', 'Chile', 'Colombia', 'Perú', 'Ecuador', 'Argentina']))
    Media_type = st.selectbox(    'Media_type',    (['Search','Social', 'Unknown', 'Display']))
    Traffic_source = st.selectbox(    'Traffic_source',    (['Google','Facebook',  'Other', 'LinkedIn']))
    dict_client = {'HN': 'Hughesnet', 'BR': 'Braun', 'EP': 'Enterprise', 'QQ':'QuickQuack', 'CJ':'ChefJames','OG':'OldGlory', 'AV':'AOV','Nuevo':'Nuevo'}


    
    Tipo_Search = st.selectbox(    'Industria',    (["Home - Home Improvement", "Animals - Pets",    "Apparel - Fashion",    "Arts - Entertainment",    "Attorneys - Legal Services",    "Automotive - For sale",
                                                       "Automotive - Repair, Service and Parts",    "Beauty - Personal Care",    "Business Services",    "Career - Employment",
                                                       "Dentists - Dental Services",    "Education - Instruction",    "Finance - Insurance",    "Furniture",    "Health - Fitness",
                                                       "Industrial - Commercial",    "Personal Services",    "Physicians - Surgeons",    "Real Estate",
                                                       "Restaurants - Food",    "Shopping, Collectibles and gifts",    "Sports and Recreation", "Travel", "Telecommunications", "Technology", "Electronics",    "Ninguna de las anteriores"     ]))
   
    Tipo_FB = df_dicc_industrias.loc[Tipo_Search]['Facebook']
   
    Tipo_YT = df_dicc_industrias.loc[Tipo_Search]['Youtube']

    
    Client = st.selectbox(    'Client',    (['BR','HN',  'EP', 'QQ', 'CJ','OG', 'AV','Nuevo'])) #['Hughesnet', 'Braun', 'Enterprise', 'QuickQuack', 'ChefJames','OldGlory', 'AOV']
    Client = dict_client[Client]
    
    Format_New = st.selectbox(    'Format_New',    (['Display', 'Video','Multiple']))
    if Format_New == 'Multiple':
        Format_New_corregido = 'Video'
    else:
        Format_New_corregido = Format_New

    
    Platform = st.selectbox(    'Platform',    (['Google Ads Search','Meta','Google Ads Discovery','Google Ads PMAX',
                                                 'Programmatic','Linkedin','Google Ads Display','NoPlatform']))
    Strategy = st.selectbox(    'Strategy',    (['Consideration','Awareness', 'Conversion',  'Views', 'NoStrategy']))
    Plataforma = st.selectbox(    'Plataforma',    (['Google Ads','Meta',  'External Source', 'NoPlataforma']))
    Campaign_Type = st.selectbox(    'Campaign_Type',    (['SEARCH','PAGE_LIKES', 'DISCOVERY', 'OUTCOME_LEADS', 'CONVERSIONS','LINK_CLICKS', 'PERFORMANCE_MAX',  'OUTCOME_AWARENESS',
                                                           'REACH', 'OUTCOME_SALES', 'NoType', 'DISPLAY','OUTCOME_ENGAGEMENT']))
    Ecommerce = st.selectbox(    'Ecommerce',    (['Si','No']))
    Service_Product = st.selectbox(    'Service_Product',    (['Serv','Prod']))

    new_data = pd.DataFrame({
    'Año': [Año],
    'Mes': [Mes],
    'Objective': [Objective],
    'Cost': [7*Cost_periodo/dias_campaña],
    'Country': [Country],
    'Media_type': [Media_type],
    'Traffic_source': [Traffic_source],
    'Client': [Client],
    'Format_New': [Format_New_corregido],
    'Platform': [Platform],
    'Strategy': [Strategy],
    'Plataforma': [Plataforma],
    'Campaign_Type': [Campaign_Type],
    'Ecommerce': [Ecommerce],
    'Service_Product': [Service_Product],
    'Bench Gral CPC': df_search_bench.loc["Ninguna de las anteriores", "Bench Search CPC"],
    'Bench Search CPC': df_search_bench.loc[Tipo_Search, "Bench Search CPC"],
    'Bench GralSch CPL': df_search_bench.loc["Ninguna de las anteriores", "Bench Search CPL"],
    'Bench Search CPL': df_search_bench.loc[Tipo_Search, "Bench Search CPL"],
    'Bench GralSch CTR': df_search_bench.loc["Ninguna de las anteriores", "Bench Search CTR"],
    'Bench Search CTR': df_search_bench.loc[Tipo_Search, "Bench Search CTR"],
    'Bench GralSch CR': df_search_bench.loc["Ninguna de las anteriores", "Bench Search AvgCR"],
    'Bench Search AvgCR': df_search_bench.loc[Tipo_Search, "Bench Search AvgCR"],
    'Bench GralFB CPC': df_FB_bench.loc["Ninguna de las anteriores", "Bench FB CPC"],
    'Bench FB CPC': df_FB_bench.loc[Tipo_FB, "Bench FB CPC"],
    'Bench GralFB CPAction': df_FB_bench.loc["Ninguna de las anteriores", "Bench FB CPAction"],
    'Bench FB CPAction': df_FB_bench.loc[Tipo_FB, "Bench FB CPAction"],
    'Bench GralFB CTR': df_FB_bench.loc["Ninguna de las anteriores", "Bench FB CTR"],
    'Bench FB CTR': df_FB_bench.loc[Tipo_FB, "Bench FB CTR"],
    'Bench GralFB CR': df_FB_bench.loc["Ninguna de las anteriores", "Bench FB AvgCR"],
    'Bench FB AvgCR': df_FB_bench.loc[Tipo_FB, "Bench FB AvgCR"],
    'Bench GralYT CPV': df_YT_bench.loc["Ninguna de las anteriores","Bench YT CPV"],
    'Bench YT CPV': df_YT_bench.loc[Tipo_YT, "Bench YT CPV"],
    'Bench GralYT CTR': df_YT_bench.loc["Ninguna de las anteriores", "Bench YT CTR"],
    'Bench YT CTR': df_YT_bench.loc[Tipo_YT, "Bench YT CTR"],
    'Bench GralYT VR': df_YT_bench.loc["Ninguna de las anteriores", "Bench FB AvgVR"],
    'Bench FB AvgVR': df_YT_bench.loc[Tipo_YT, "Bench FB AvgVR"]
    })
    

    # Preprocesamiento de variables categóricas
    X = pd.get_dummies(new_data, columns=categorical_features)
    
    # Asegurarte de que 'new_data_encoded' tenga las mismas columnas que se utilizaron durante el entrenamiento
    for col in variables_modelo:
        if col not in X.columns:
            X[col] = False  # Agregar la columna faltante con valores predeterminados si es necesario  

    X = X[variables_modelo]
    X.columns = [str(i) for i in X.columns]


def prediccion_modelo(modelo,X):
    return modelo.predict(X)

bin_density = 300 #st.slider('Bins', min_value=250, max_value=350, step=5, value=300)

with st.expander("Input de la predicción"):
    st.write('Fecha Inicio: '+str(fecha_inicio))
    st.write('Fecha Fin: '+str(fecha_fin))
    st.write('Duración campaña [días]: '+str(dias_campaña))
    st.write('Costo total [USD]: '+str(Cost_periodo))
    st.write('Costo diario [USD/día]: '+str(round(Cost_periodo/dias_campaña)))
    st.write()
    
    st.write('Pais: '+Country)
    st.write('Media_type: '+Media_type)
    st.write('Traffic_source: '+Traffic_source)
    st.write('Client: '+Client)
    st.write('Format_New: '+Format_New)
    st.write('Platform: '+Platform)
    st.write('Strategy: '+Strategy)
    st.write('Plataforma: '+Plataforma)
    st.write('Campaign_Type: '+Campaign_Type)
    st.write('Ecommerce: '+Ecommerce)
    st.write('Service_Product: '+Service_Product)
    st.write('Industria: '+Tipo_Search)





st.button("Reset", type="primary")
if st.button('Hacer predicción'):



    def histo(df,metrica,valor,bins=bin_density, intervalo = 0):
        chart = alt.Chart(df).mark_bar(
        opacity=0.3,
        binSpacing=0
    ).encode(
        alt.X(metrica+':Q').bin(maxbins=bin_density),
        alt.Y('count()').stack(None),            
    ).properties(
            width=1000,
            height=600
        ).interactive()

        linea_valor = alt.Chart(pd.DataFrame({'valor_linea': [valor]})).mark_rule(color='red').encode(
    x='valor_linea:Q',
    size=alt.value(2))  # Grosor de la línea)

        linea1 = alt.Chart(pd.DataFrame({'valor_linea': [max(valor - intervalo,0)]})).mark_rule(color='pink').encode(
            x='valor_linea:Q',
            size=alt.value(2))  # Grosor de la línea)

        linea2 = alt.Chart(pd.DataFrame({'valor_linea': [valor + intervalo]})).mark_rule(color='pink').encode(
            x='valor_linea:Q',
            size=alt.value(2))  # Grosor de la línea)

       
        return chart + linea_valor + linea1 + linea2

    
    
    
        
    X_CPC = X.copy()
    X_CPM = X.copy()
    X_CTR = X.copy()
    X_CPV = X.copy()
    


### CPC


    intervalo = 0.258

    
    X_Scaled = loaded_scaler_CPC.transform(X_CPC[['Año','Mes','Cost','Bench Gral CPC','Bench Search CPC', 'Bench GralSch CPL', 'Bench Search CPL','Bench GralSch CTR', 'Bench Search CTR', 'Bench GralSch CR',
                                          'Bench Search AvgCR','Bench GralFB CPC', 'Bench FB CPC','Bench GralFB CPAction', 'Bench FB CPAction', 'Bench GralFB CTR','Bench FB CTR',
                                          'Bench GralFB CR', 'Bench FB AvgCR',       'Bench GralYT CPV', 'Bench YT CPV', 'Bench GralYT CTR', 'Bench YT CTR','Bench GralYT VR', 'Bench FB AvgVR']])
    X_pca = loaded_pca_CPC.transform(X_Scaled)
    X_pca = pd.DataFrame(X_pca)
    X_CPC['X_pca_0'] = X_pca[0]
    X_CPC['X_pca_1'] = X_pca[1]

    X_NN_CPC = loaded_scaler_NN_CPC.transform(X_CPC)
  
    if Client == 'Nuevo':

    

        
        variables_modelo_sinClient = xgboost_CPC_sinClient.feature_names_in_
        X_NN_CPC_sinClient = loaded_scaler_NN_CPC_sinClient.transform(X_CPC[variables_modelo_sinClient])
        pred_CPC = (  max(0,prediccion_modelo(xgboost_CPC_sinClient,X_CPC[variables_modelo_sinClient])[0]) + max(0,prediccion_modelo(NN_CPC_sinClient,X_NN_CPC_sinClient)[0]) + max(0,prediccion_modelo(RF_CPC_sinClient,X_CPC[variables_modelo_sinClient])[0]) ) / 3
        st.write('Cliente Nuevo')
    else:
        pred_CPC = (  max(0,prediccion_modelo(xgboost_CPC,X_CPC)[0]) + max(0,prediccion_modelo(NN_CPC,X_NN_CPC)[0]) + max(0,prediccion_modelo(RF_CPC,X_CPC)[0]) ) / 3

    st.write('CPC')
    #st.write('XGBoost')
    #st.write(prediccion_modelo(xgboost_CPC,X_CPC)[0])
    #st.write('Redes Neuronales')
    #st.write(prediccion_modelo(NN_CPC,X_NN_CPC)[0])
    #st.write('Random Forest')
    #st.write(prediccion_modelo(RF_CPC,X_CPC)[0])     
    #st.write('Ensamble')
    st.write(round(pred_CPC,3))
    st.write('En el  Intervalo: ['+str(round(max(0,pred_CPC-intervalo),3))+' - '+str(round(pred_CPC+intervalo,3))+'] con un nivel de confianza del 85%')
    st.altair_chart(histo(df,'CPC',pred_CPC, intervalo = intervalo), use_container_width=False, theme=None)









### CPM







    intervalo = 54

    X_Scaled = loaded_scaler_CPM.transform(X_CPM[['Año','Mes','Cost','Bench Gral CPC','Bench Search CPC', 'Bench GralSch CPL', 'Bench Search CPL','Bench GralSch CTR', 'Bench Search CTR', 'Bench GralSch CR',
                                          'Bench Search AvgCR','Bench GralFB CPC', 'Bench FB CPC','Bench GralFB CPAction', 'Bench FB CPAction', 'Bench GralFB CTR','Bench FB CTR',
                                          'Bench GralFB CR', 'Bench FB AvgCR',       'Bench GralYT CPV', 'Bench YT CPV', 'Bench GralYT CTR', 'Bench YT CTR','Bench GralYT VR', 'Bench FB AvgVR']])
    X_pca = loaded_pca_CPM.transform(X_Scaled)
    X_pca = pd.DataFrame(X_pca)
    X_CPM['X_pca_0'] = X_pca[0]
    X_CPM['X_pca_1'] = X_pca[1]    


    X_NN_CPM = loaded_scaler_NN_CPM.transform(X_CPM)
    if Client == 'Nuevo':

    

        
        variables_modelo_sinClient = xgboost_CPM_sinClient.feature_names_in_
        X_NN_CPM_sinClient = loaded_scaler_NN_CPM_sinClient.transform(X_CPM[variables_modelo_sinClient])
        pred_CPM = (  max(0,prediccion_modelo(xgboost_CPM_sinClient,X_CPM[variables_modelo_sinClient])[0]) + max(0,prediccion_modelo(NN_CPM_sinClient,X_NN_CPM_sinClient)[0]) + max(0,prediccion_modelo(RF_CPM_sinClient,X_CPM[variables_modelo_sinClient])[0]) ) / 3
        st.write('Cliente Nuevo')
    else:    
        pred_CPM = (  max(0,prediccion_modelo(xgboost_CPM,X_CPM)[0]) + max(0,prediccion_modelo(NN_CPM,X_NN_CPM)[0]) + max(0,prediccion_modelo(RF_CPM,X_CPM)[0]) ) / 3


    st.write('CPM')
    #st.write('XGBoost')
    #st.write(prediccion_modelo(xgboost_CPM,X_CPM)[0])
   # st.write('Redes Neuronales')
   # st.write(prediccion_modelo(NN_CPM,X_NN_CPM)[0])
    #st.write('Random Forest')
    #st.write(prediccion_modelo(RF_CPM,X_CPM)[0])     
    #st.write('Ensamble')
    st.write(round(pred_CPM,3))
    st.write('En el  Intervalo: ['+str(round(max(0,pred_CPM-intervalo),3))+' - '+str(round(pred_CPM+intervalo,3))+'] con un nivel de confianza del 85%')
    st.altair_chart(histo(df,'CPM',pred_CPM, intervalo = intervalo), use_container_width=False, theme=None)










### CTR





    
    intervalo = 0.0568

    X_Scaled = loaded_scaler_CTR.transform(X_CTR[['Año','Mes','Cost','Bench Gral CPC','Bench Search CPC', 'Bench GralSch CPL', 'Bench Search CPL','Bench GralSch CTR', 'Bench Search CTR', 'Bench GralSch CR',
                                      'Bench Search AvgCR','Bench GralFB CPC', 'Bench FB CPC','Bench GralFB CPAction', 'Bench FB CPAction', 'Bench GralFB CTR','Bench FB CTR',
                                      'Bench GralFB CR', 'Bench FB AvgCR',       'Bench GralYT CPV', 'Bench YT CPV', 'Bench GralYT CTR', 'Bench YT CTR','Bench GralYT VR', 'Bench FB AvgVR']])
    X_pca = loaded_pca_CTR.transform(X_Scaled)
    X_pca = pd.DataFrame(X_pca)
    X_CTR['X_pca_0'] = X_pca[0]
    X_CTR['X_pca_1'] = X_pca[1]


    X_NN_CTR = loaded_scaler_NN_CTR.transform(X_CTR)

    if Client == 'Nuevo':

    

        
        variables_modelo_sinClient = xgboost_CTR_sinClient.feature_names_in_
        X_NN_CTR_sinClient = loaded_scaler_NN_CTR_sinClient.transform(X_CTR[variables_modelo_sinClient])
        pred_CTR = (  max(0,prediccion_modelo(xgboost_CTR_sinClient,X_CTR[variables_modelo_sinClient])[0]) + max(0,prediccion_modelo(NN_CTR_sinClient,X_NN_CTR_sinClient)[0]) + max(0,prediccion_modelo(RF_CTR_sinClient,X_CTR[variables_modelo_sinClient])[0]) ) / 3
        st.write('Cliente Nuevo')
    else:   
        pred_CTR = (  max(0,prediccion_modelo(xgboost_CTR,X_CTR)[0]) + max(0,prediccion_modelo(NN_CTR,X_NN_CTR)[0]) + max(0,prediccion_modelo(RF_CTR,X_CTR)[0]) ) / 3

    st.write('CTR')
    #st.write('XGBoost')
    #st.write(prediccion_modelo(xgboost_CTR,X_CTR)[0])
    #st.write('Redes Neuronales')
    #st.write(prediccion_modelo(NN_CTR,X_NN_CTR)[0])
    #st.write('Random Forest')
    #st.write(prediccion_modelo(RF_CTR,X_CTR)[0])     
    #st.write('Ensamble')
    st.write(round(pred_CTR,3))
    st.write('En el  Intervalo: ['+str(round(max(0,pred_CTR-intervalo),3))+' - '+str(round(pred_CTR+intervalo,3))+'] con un nivel de confianza del 85%')
    st.altair_chart(histo(df,'CTR',pred_CTR, intervalo = intervalo), use_container_width=False, theme=None)





    
#CPV
    X_CPV = X_CPV[loaded_scaler_NN_CPV.feature_names_in_]
    for col in RF_CPV.feature_names_in_:
        if col not in X_CPV.columns:
            X_CPV[col] = False  # Agregar la columna faltante con valores predeterminados si es necesario
            print(col)

    intervalo = 12.8
   
    
    X_Scaled = loaded_scaler_CPV.transform(X_CPV[['Año','Mes','Cost','Bench Gral CPC','Bench Search CPC', 'Bench GralSch CPL', 'Bench Search CPL','Bench GralSch CTR', 'Bench Search CTR', 'Bench GralSch CR',
                                          'Bench Search AvgCR','Bench GralFB CPC', 'Bench FB CPC','Bench GralFB CPAction', 'Bench FB CPAction', 'Bench GralFB CTR','Bench FB CTR',
                                          'Bench GralFB CR', 'Bench FB AvgCR', 'Bench GralYT CPV', 'Bench YT CPV', 'Bench GralYT CTR', 'Bench YT CTR','Bench GralYT VR', 'Bench FB AvgVR']])
    X_pca = loaded_pca_CPV.transform(X_Scaled)
    X_pca = pd.DataFrame(X_pca)
    X_CPV['X_pca_0'] = X_pca[0]
    X_CPV['X_pca_1'] = X_pca[1]

    X_CPV = X_CPV[loaded_scaler_NN_CPV.feature_names_in_]
    #X_CPV = X_CPV[RF_CPV.feature_names_in_]
    X_CPV.columns = [str(i) for i in X_CPV.columns]
 
    X_NN_CPV = loaded_scaler_NN_CPV.transform(X_CPV)

    if Format_New_corregido == 'Video':
        st.write('CPV')
        #st.write('XGBoost')
        #st.write(prediccion_modelo(xgboost_CPV,X_CPV)[0])
        #st.write('Redes Neuronales')
        #st.write(prediccion_modelo(NN_CPV,X_NN_CPV)[0])
        #st.write('Random Forest')
        #st.write(prediccion_modelo(RF_CPV,X_CPV)[0])     
        
        pred_RF_CPV = prediccion_modelo(RF_CPV,X_CPV)[0]
        pred_XGB_CPV = prediccion_modelo(xgboost_CPV,X_CPV)[0]
        pred_NN_CPV = prediccion_modelo(NN_CPV,X_NN_CPV)[0]
        if pred_NN_CPV > 3:
                pred_NN_CPV = (pred_RF_CPV + pred_XGB_CPV)/2
        elif pred_NN_CPV < 0:
                pred_NN_CPV = (pred_RF_CPV + pred_XGB_CPV)/2
        
        pred_CPV = (  pred_XGB_CPV + pred_NN_CPV + pred_RF_CPV ) / 3
        #st.write('Ensamble')
        st.write(round(pred_CPV,3))
        st.write('En el  Intervalo: ['+str(round(max(0,pred_CPV-intervalo),3))+' - '+str(round(pred_CPV+intervalo,3))+'] con un nivel de confianza del 85%')
        st.altair_chart(histo(df_cpv,'CPV',pred_CPV, intervalo = intervalo), use_container_width=False, theme=None)



        
else:
    st.write('Prepara tu predicción')
    
