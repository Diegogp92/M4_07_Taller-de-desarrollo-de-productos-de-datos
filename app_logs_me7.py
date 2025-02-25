# Cargamos las librerías que consideremos necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
import xgboost as xgb

st.title("Modelo de Temperatura de escape en el catalizador de un vehículo ICE")
st.header("En esta práctica veremos cómo predecir la temperatura de escape en función de otras variables. Para ello, necesitaremos cargar un dataset con la siguiente información:")

# Cargamos la descripción de las variables que estudiaremos
df_variables = pd.read_csv('https://raw.github.com/Diegogp92/M4_07_Taller-de-desarrollo-de-productos-de-datos/main/descripcion_variables.csv', encoding='latin1')
df_variables

# Pido un drop del log de me7logger por pantalla en streamlit (con la intención de usar el archivo logs_me7logger.csv)
archivo = st.file_uploader("Arrastra aquí tus logs en formato .CSV", type="csv")

# Verificar si se cargó un archivo
if archivo is not None:
    st.success("Archivo subido correctamente.")
    # Leer archivos CSV o Excel
    try:
        if archivo.name.endswith(".csv"):
            df_log = pd.read_csv(archivo)
        else:
            df_log = None
            st.warning("Formato no soportado.")
        # Mostrar el DataFrame
        if df_log is not None:
            st.write("Vista previa del archivo (tikatm_w se ha pasado a Kelvin para ser más rigurososo con los cálculos):")
            df_log['tikatm_w'] = df_log['tikatm_w']+273.15 # Para pasarlo a Kelvin
            st.dataframe(df_log.head())

    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

if df_log is not None:
    st.header("EDA inicial:")
    st.write('Hacemos un pequeño estudio de los datos aportados')
    st.dataframe(df_log.describe())
    #print(df_log.info())

    st.write('Correlación entre variables')
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_log.corr(), annot=True, cmap='coolwarm', cbar = False, fmt='.2f', linewidths=0.5)
    #plt.title("Heatmap de Correlación")
    st.pyplot(plt)
    #plt.show()

    st.write('Puede que los datos estén desbalanceados. Aquí lo veremos en función de la carga (Variable primaria de la bosch me7.5):')
    # Distribución de los datos por niveles de carga
    # Definición de bins
    bins = np.arange(0, (max(df_log['rlsol_w']) // 20 + 1) * 20 + 20, 20).astype(np.int64) # Para que se amolde a los datos y siempre coja múltiplos de 20
    df_distribucion_datos = df_log.copy()
    df_distribucion_datos['rlsol_w_bin'] = pd.cut(df_distribucion_datos['rlsol_w'], bins, right=False)
    # Cuento el número de líneas por cada bin
    bin_counts = df_distribucion_datos['rlsol_w_bin'].value_counts().sort_index()

    # Creo el gráfico
    plt.figure(figsize=(10, 5))
    sns.barplot(x=bin_counts.index.astype(str), y=bin_counts.values)
    plt.xlabel("Rangos de rlsol_w")
    plt.xticks(rotation=45)
    plt.ylabel("Número de datos")
    plt.title("Distribución de los datos")
    #plt.show()
    st.pyplot(plt)

    st.header("EDA Dinámico:")

    # Crear un slider con rango
    corte_inf_carga, corte_sup_carga = st.slider(
        "Selecciona el rango de valores a estudiar",
        min_value=0, max_value=190, value=(0, 190)  # Inicialización deslimitado
    )

    st.dataframe(df_log.loc[(df_log['rlsol_w'] >= corte_inf_carga) & (df_log['rlsol_w'] <= corte_sup_carga)].describe())
    
    # Crear boxplots separados para cada variable con los datos acotados
    # Mostrarlos juntos
    num_columns = len(df_log.columns)
    fig, axes = plt.subplots(1, num_columns, figsize=(20, 5))  # 1 fila, num_columns columnas

    # Ajustar el espacio entre los subplots
    plt.subplots_adjust(wspace=0.5)

    # Crear boxplots separados para cada variable
    for i, column in enumerate(df_log.columns):
        sns.boxplot(df_log.loc[(df_log['rlsol_w'] >= corte_inf_carga) & (df_log['rlsol_w'] <= corte_sup_carga), column], ax=axes[i])
        axes[i].set_title(column)
        axes[i].set_ylabel('')  # Quita el título del eje Y

    # Mostrar la figura
    st.write("Boxplots de todas las variables con los datos acotados:")
    st.pyplot(plt)

    st.header("Modelos")
    st.write("A continuación se entrenarán 3 modelos con los datos acotados: Regresión lineal, Puntos interpolados linealmente y XGBoost.")
    st.write("Es esperable que obtengamos los mejores resultados con XGBoost, pero intentaremos alcanzar buenos resultados con los 2 primeros para que su implementación en microcontroladores y ejecución en tiempo real sea viable.")
    st.write("Su entrenamiento tendrá lugar en combinación con validaciones cruzadas y búsqueda de los mejores hiperparámetros para adecuarse lo mejor posible al rango seleccionado arriba.")
    st.write("El entrenamiento comenzará cuando pulses el botón. Por favor, ten paciencia, puede tardar entre 2 y 5 minutos dependiendo de tu HW.")

    # Botón para iniciar el entrenamiento
    if st.button("Entrenar Modelos"):
        with st.spinner("Procesando..."):
            # Hago los folds para las validaciones cruzadas
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Designación de variables predictivas y predictoras
            X = (df_log.loc[(df_log['rlsol_w'] >= corte_inf_carga) & (df_log['rlsol_w'] <= corte_sup_carga)]).drop(columns=['tikatm_w'])
            y = df_log.loc[(df_log['rlsol_w'] >= corte_inf_carga) & (df_log['rlsol_w'] <= corte_sup_carga)]['tikatm_w']

            # Regresión 
            modelo_reg = LinearRegression()
            modelo_reg.fit(X, y)
            y_pred_reg = modelo_reg.predict(X)

            # Calculo métricas de error
            r2_reg = r2_score(y, y_pred_reg)
            mse_reg = mean_squared_error(y, y_pred_reg)
            me_reg = mse_reg**(1/2)

            # Imprimo resultados
            st.write("Con la Regresión lineal hemos obtenido las siguientes métricas de error:")
            st.write(f"R²: {r2_reg:.4f}")
            st.write(f"MSE: {mse_reg:.2f}°C²")
            st.write(f"Error medio: {me_reg:.2f}°C")

            # Modelo por puntos (nodos) e interpolación lineal entre ellos (splines de grado 1)
            # Multipunto: mtp

            # Definir el rango de exploración de hiperparámetros
            param_grid_mtp = {
                'spline__degree': [1],
                'spline__n_knots': range(1, 51, 2)
            }

            # Crear un pipeline que primero transforme los datos con splines y luego aplique una regresión lineal
            modelo_pipeline_mtp = Pipeline([
                ('spline', SplineTransformer()),
                ('regression', LinearRegression())
            ])

            # Configurar la búsqueda de hiperparámetros
            grid_search_mtp = GridSearchCV(
                estimator=modelo_pipeline_mtp,
                param_grid=param_grid_mtp,
                cv=kf,
                scoring="r2",
                n_jobs=-1,  # para usar todos los núcleos
            )

            # Ajustar el modelo a los datos
            grid_search_mtp.fit(X, y)

            # Imprimir los mejores parámetros y el R²
            st.write("Con la Multipunto + interpolación lineal estos son los mejores resultados:")
            st.write(f"Mejores hiperparámetros: {grid_search_mtp.best_params_}")
            st.write(f"R²: {grid_search_mtp.best_score_:.4f}")

            # Asignar los hiperparámetros óptimos para futuros cálculos
            modelo_mtp = grid_search_mtp.best_estimator_

            # Calcular errores más intuitivos
            y_pred_cv_mtp = cross_val_predict(modelo_mtp, X, y, cv=kf)
            mse_mtp = mean_squared_error(y, y_pred_cv_mtp)
            me_mtp = mse_mtp**(1/2)

            # Imprimir resultados
            st.write(f"MSE: {mse_mtp:.2f}°C²")
            st.write(f"Error medio: {me_mtp:.2f}°C")

            # XGBoost
            modelo_xgb = xgb.XGBRegressor()

            # Rango de exploración hiperparámetros
            param_grid_xgb = {
                'max_depth': [6,8,10,12],
                'learning_rate': [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.20, 0.25, 0.30],
                'n_estimators': [100, 150, 175],
            }

            grid_search_xgb = GridSearchCV(
                estimator=modelo_xgb,
                param_grid=param_grid_xgb,
                cv=kf,
                scoring="r2",
                n_jobs=-1, #para usar todos los núcleos
                )

            grid_search_xgb.fit(X, y)
            st.write("Con el XGBoost esto son los mejores resultados:")
            st.write(f"Mejores hiperparámetros: {grid_search_xgb.best_params_}")
            st.write(f"R²: {grid_search_xgb.best_score_:.4f}")

            # Asigno los hiperparámetros óptimos para futuros cálculos
            modelo_xgb = grid_search_xgb.best_estimator_

            # Calculo errores más intuitivos
            y_pred_cv_xgb = cross_val_predict(modelo_xgb, X, y, cv=kf)
            mse_xgb = mean_squared_error(y, y_pred_cv_xgb)
            me_xgb = mse_xgb**(1/2)

            # Imprimo resultados
            st.write(f"MSE: {mse_xgb:.2f}°C²")
            st.write(f"Error medio: {me_xgb:.2f}°C")

            #Predicciones
            # Añado las predicciones como nuevas columnas a df_log_pred, copia de df_log
            df_log_pred = df_log.copy()
            features = [col for col in df_log_pred.columns if col not in ['tikatm_w', 'tikatm_reg', 'tikatm_mtp', 'tikatm_xgb']]
            df_log_pred['tikatm_reg'] = df_log_pred[features].apply(lambda row: modelo_reg.predict([row.values])[0], axis=1)
            df_log_pred['tikatm_mtp'] = df_log_pred[features].apply(lambda row: modelo_mtp.predict([row.values])[0], axis=1)
            df_log_pred['tikatm_xgb'] = df_log_pred[features].apply(lambda row: modelo_xgb.predict([row.values])[0], axis=1)

            # Creo columnas con errores relativos
            df_log_pred['error_rel_reg'] = abs((df_log_pred['tikatm_w'] - df_log_pred['tikatm_reg'])/(df_log_pred['tikatm_w']))*100
            df_log_pred['error_rel_mtp'] = abs((df_log_pred['tikatm_w'] - df_log_pred['tikatm_mtp'])/(df_log_pred['tikatm_w']))*100
            df_log_pred['error_rel_xgb'] = abs((df_log_pred['tikatm_w'] - df_log_pred['tikatm_xgb'])/(df_log_pred['tikatm_w']))*100

            # Creo columnas con errores absolutos
            df_log_pred['error_abs_reg'] = abs(df_log_pred['tikatm_w'] - df_log_pred['tikatm_reg'])
            df_log_pred['error_abs_mtp'] = abs(df_log_pred['tikatm_w'] - df_log_pred['tikatm_mtp'])
            df_log_pred['error_abs_xgb'] = abs(df_log_pred['tikatm_w'] - df_log_pred['tikatm_xgb'])

            # %% [markdown]
            # ## Errores

            # %%
            # Definir los bins para discretizar rlsol_w y nmot_w
            rlsol_bins = np.arange(0, 190+10, 10)
            nmot_bins = np.arange(0, 6500+500, 500)

            # Copiar df_log_pred para no afectarlo con los bins
            df_bins = df_log_pred.copy()

            # Asignar cada valor de rlsol_w y nmot_w a un bin
            df_bins['rlsol_bin'] = pd.cut(df_bins['rlsol_w'], bins=rlsol_bins, labels=np.round(rlsol_bins[:-1], 2))
            df_bins['nmot_bin'] = pd.cut(df_bins['nmot_w'], bins=nmot_bins, labels=np.round(nmot_bins[:-1], 2))

            # Colormap personalizado
            colors = ["green", "yellow", "red"]
            n_bins = 100  # Número de segmentos de color
            cmap_name = "green_yellow_red"
            custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

            # %%
            st.header("Puesta en contexto")
            st.write("tikatk_w en f(nmot y rlsol_w)")

            # Crear la tabla pivote con el error relativo promedio
            heatmap_tikatm = df_bins.pivot_table(
                index='nmot_bin', 
                columns='rlsol_bin', 
                values='tikatm_w',
                aggfunc=np.mean
            )

            # Convertir los índices a valores numéricos para mejor visualización
            heatmap_tikatm.index = heatmap_tikatm.index.astype(np.int64)
            heatmap_tikatm.columns = heatmap_tikatm.columns.astype(np.int64)

            annot_tikatm = heatmap_tikatm.copy().astype(str)
            for i in range(heatmap_tikatm.shape[0]):
                for j in range(heatmap_tikatm.shape[1]):
                        # Convertir de Kelvin a Celsius
                        heatmap_tikatm.iloc[i, j] = heatmap_tikatm.iloc[i, j] - 273.15
                        annot_tikatm.iloc[i, j] = f"{heatmap_tikatm.iloc[i, j]:.1f}°C"

            # Crear el heatmap
            plt.figure(figsize=(16, 6))
            sns.heatmap(heatmap_tikatm, cmap=custom_cmap, cbar=False, annot=annot_tikatm, fmt='', linewidths=0.5)

            # Añadir título y etiquetas
            plt.title('tikatm_w [°C]')
            plt.xlabel('Carga del Motor Especificada [%] (rlsol_w)')
            plt.ylabel('Velocidad del Motor (nmot_w)')

            plt.gca().invert_yaxis() # Poner el origen del eje Y abajo

            # Ajustar las etiquetas del eje X a la izquierda
            plt.xticks(
                ticks=np.arange(len(heatmap_tikatm.columns)) + 0,
                labels=heatmap_tikatm.columns,
                rotation=0,
            )

            # Ajustar las etiquetas del eje y abajo
            plt.yticks(
                ticks=np.arange(len(heatmap_tikatm.index)) + 0,
                labels=heatmap_tikatm.index,
                rotation=0,
            )

            # Mostrar el gráfico
            st.pyplot(plt)
            st.write("Las celdas rojas en cargas bajas aparentemente no tienen sentido.")
            st.write("Por qué ese cambio tan abrupto en 2500rpm? Y por qué no hay muestras para esas rpm en la segunda columna de carga?")
            st.write("Los ensayos que típicamente se hacen cuando se modifica el SW de control de motor de un coche, consisten en lanzadas en 3ª marcha con el acelerador a fondo desde unas 2500rpm hasta la zona más alta del tacómetro.")
            st.write("Cuando se llega, se suelta el acelerador y se deja el coche reducir de velocidad en punto muerto para medir las pérdidas mecánicas de potencia.")
            st.write("Con lo cual, la carga solicitada va a bajar repentinamente y las rpm van a disminuir rápidamente a ralentí, pero viniendo de una situación de estrés máximo y las temperaturas del gas y elementos de escape seguirán siendo muy altas.")
            st.write("Entre 500 y 2500rpm ya apreciamos temperaturas más moderadas debido a que hay mucho dato de crucero.")
            st.write("Para hacer un modelo de temperatura de escape, deberían haberse contemplado más variables y hacer ensayos manteniendo cada variable fija mientras varía el resto, para tener toda la zona del gráfico completa y con una densidad homogénea de datos")
            st.write("Pero para este proyecto, tiraremos con el dataset que tenemos.")

            st.header("Errores de Regresión")
            st.write("En gris: Predicciones fuera del área de entrenamiento")

            # Crear la tabla pivote con el error relativo promedio
            heatmap_reg_rel = df_bins.pivot_table(
                index='nmot_bin', 
                columns='rlsol_bin', 
                values='error_rel_reg',
                aggfunc=np.mean
            )

            # Crear la tabla pivote con el error absoluto promedio
            heatmap_reg_abs = df_bins.pivot_table(
                index='nmot_bin', 
                columns='rlsol_bin', 
                values='error_abs_reg',
                aggfunc=np.mean
            )

            # Convertir los índices a valores numéricos para mejor visualización
            heatmap_reg_rel.index = heatmap_reg_rel.index.astype(np.int64)
            heatmap_reg_rel.columns = heatmap_reg_rel.columns.astype(np.int64)
            heatmap_reg_abs.index = heatmap_reg_abs.index.astype(np.int64)
            heatmap_reg_abs.columns = heatmap_reg_abs.columns.astype(np.int64)

            # Combinar los datos de ambas tablas en una sola cadena de texto para cada celda
            annot_combinado_reg = heatmap_reg_rel.copy().astype(str)
            for i in range(heatmap_reg_rel.shape[0]):
                for j in range(heatmap_reg_rel.shape[1]):
                    # Aplico "Fuera de Rango"
                    if heatmap_reg_rel.iloc[i, j] > 999 or heatmap_reg_abs.iloc[i, j] > 999:
                        annot_combinado_reg.iloc[i, j] = "F/R"  # Fuera de Rango, para no ensuciar el gráfico con valores que se salen de la celda
                    else:
                        annot_combinado_reg.iloc[i, j] = f"{heatmap_reg_rel.iloc[i, j]:.2f}%\n({heatmap_reg_abs.iloc[i, j]:.1f}°C)"

            # Crear el heatmap
            plt.figure(figsize=(16, 6))
            sns.heatmap(heatmap_reg_rel, cmap=custom_cmap, cbar=False, annot=annot_combinado_reg, fmt='', linewidths=0.5)

            # Añadir título y etiquetas
            plt.title('Error Relativo (Absoluto) de Predicción por Regresión')
            plt.xlabel('Carga del Motor Especificada [%] (rlsol_w)')
            plt.ylabel('Velocidad del Motor (nmot_w)')

            # Crear la máscara para las celdas fuera del rango de estudio
            # Sólo esta vez. La reutilizaré de ahora en adelante
            mask = (heatmap_reg_rel.columns < corte_inf_carga) | (heatmap_reg_rel.columns >= corte_sup_carga)
            gray_mask = np.zeros_like(heatmap_reg_rel, dtype=bool)
            gray_mask[:, mask] = True

            # Heatmap gris
            sns.heatmap(
                heatmap_reg_rel, 
                cmap=['lightgray'],
                cbar=False, 
                annot=annot_combinado_reg, 
                fmt='', 
                linewidths=0.5,
                mask=~gray_mask,  # Invertir la máscara para aplicar el gris solo a las celdas fuera del rango
            )
            plt.gca().invert_yaxis() # Poner el origen del eje Y abajo

            # Ajustar las etiquetas del eje X a la izquierda
            plt.xticks(
                ticks=np.arange(len(heatmap_reg_rel.columns)) + 0,
                labels=heatmap_reg_rel.columns,
                rotation=0,
            )

            # Ajustar las etiquetas del eje y abajo
            plt.yticks(
                ticks=np.arange(len(heatmap_reg_rel.index)) + 0,
                labels=heatmap_reg_rel.index,
                rotation=0,
            )

            # Mostrar el gráfico
            st.pyplot(plt)

            st.header("Errores de Multipunto")
            st.write("En gris: Predicciones fuera del área de entrenamiento")
            st.write("F/R = Fuera de rango")

            # Crear la tabla pivote con el error relativo promedio
            heatmap_mtp_rel = df_bins.pivot_table(
                index='nmot_bin', 
                columns='rlsol_bin', 
                values='error_rel_mtp',
                aggfunc=np.mean
            )

            # Crear la tabla pivote con el error absoluto promedio
            heatmap_mtp_abs = df_bins.pivot_table(
                index='nmot_bin', 
                columns='rlsol_bin', 
                values='error_abs_mtp',
                aggfunc=np.mean
            )

            # Convertir los índices a valores numéricos para mejor visualización
            heatmap_mtp_rel.index = heatmap_mtp_rel.index.astype(np.int64)
            heatmap_mtp_rel.columns = heatmap_mtp_rel.columns.astype(np.int64)
            heatmap_mtp_abs.index = heatmap_mtp_abs.index.astype(np.int64)
            heatmap_mtp_abs.columns = heatmap_mtp_abs.columns.astype(np.int64)

            # Combinar los datos de ambas tablas en una sola cadena de texto para cada celda
            annot_combinado_mtp = heatmap_mtp_rel.copy().astype(str)
            for i in range(heatmap_mtp_rel.shape[0]):
                for j in range(heatmap_mtp_rel.shape[1]):
                    # Aplico "Fuera de Rango"
                    if heatmap_mtp_rel.iloc[i, j] > 999 or heatmap_mtp_abs.iloc[i, j] > 999:
                        annot_combinado_mtp.iloc[i, j] = "F/R"  # Fuera de Rango, para no ensuciar el gráfico con valores que se salen de la celda
                    else:
                        annot_combinado_mtp.iloc[i, j] = f"{heatmap_mtp_rel.iloc[i, j]:.2f}%\n({heatmap_mtp_abs.iloc[i, j]:.1f}°C)"

            # Crear el heatmap
            plt.figure(figsize=(16, 6))
            sns.heatmap(heatmap_mtp_rel, cmap=custom_cmap, cbar=False, annot=annot_combinado_mtp, fmt='', linewidths=0.5, mask=gray_mask)  # Aplicar la máscara para las celdas en gris)

            # Añadir título y etiquetas
            plt.title('Error Relativo (Absoluto) de Predicción por Multipunto')
            plt.xlabel('Carga del Motor Especificada [%] (rlsol_w)')
            plt.ylabel('Velocidad del Motor (nmot_w)')

            # Heatmap gris
            sns.heatmap(
                heatmap_reg_rel, 
                cmap=['lightgray'],
                cbar=False, 
                annot=annot_combinado_mtp, 
                fmt='', 
                linewidths=0.5,
                mask=~gray_mask,  # Invertir la máscara para aplicar el gris solo a las celdas fuera del rango
            )
            plt.gca().invert_yaxis() # Poner el origen del eje Y abajo

            # Ajustar las etiquetas del eje X a la izquierda
            plt.xticks(
                ticks=np.arange(len(heatmap_mtp_rel.columns)) + 0,
                labels=heatmap_mtp_rel.columns, 
                rotation=0,
            )

            # Ajustar las etiquetas del eje y abajo
            plt.yticks(
                ticks=np.arange(len(heatmap_mtp_rel.index)) + 0,
                labels=heatmap_mtp_rel.index,
                rotation=0,
            )

            # Mostrar el gráfico
            st.pyplot(plt)

            st.header("Errores de XGBoost")
            st.write("En gris: Predicciones fuera del área de entrenamiento")

            # Crear la tabla pivote con el error relativo promedio
            heatmap_xgb_rel = df_bins.pivot_table(
                index='nmot_bin', 
                columns='rlsol_bin', 
                values='error_rel_xgb',
                aggfunc=np.mean
            )

            # Crear la tabla pivote con el error absoluto promedio
            heatmap_xgb_abs = df_bins.pivot_table(
                index='nmot_bin', 
                columns='rlsol_bin', 
                values='error_abs_xgb',
                aggfunc=np.mean
            )

            # Convertir los índices a valores numéricos para mejor visualización
            heatmap_xgb_rel.index = heatmap_xgb_rel.index.astype(np.int64)
            heatmap_xgb_rel.columns = heatmap_xgb_rel.columns.astype(np.int64)
            heatmap_xgb_abs.index = heatmap_xgb_abs.index.astype(np.int64)
            heatmap_xgb_abs.columns = heatmap_xgb_abs.columns.astype(np.int64)

            # Combinar los datos de ambas tablas en una sola cadena de texto para cada celda
            annot_combinado_xgb = heatmap_xgb_rel.copy().astype(str)
            for i in range(heatmap_xgb_rel.shape[0]):
                for j in range(heatmap_xgb_rel.shape[1]):
                    # Aplico "Fuera de Rango"
                    if heatmap_xgb_rel.iloc[i, j] > 999 or heatmap_xgb_abs.iloc[i, j] > 999:
                        annot_combinado_xgb.iloc[i, j] = "F/R" # Fuera de Rango, para no ensuciar el gráfico con valores que se salen de la celda
                    else:
                        annot_combinado_xgb.iloc[i, j] = f"{heatmap_xgb_rel.iloc[i, j]:.2f}%\n({heatmap_xgb_abs.iloc[i, j]:.1f}°C)"

            # Crear el heatmap
            plt.figure(figsize=(16, 6))
            sns.heatmap(heatmap_xgb_rel, cmap=custom_cmap, cbar=False, annot=annot_combinado_xgb, fmt='', linewidths=0.5, mask=gray_mask)  # Aplicar la máscara para las celdas en gris)

            # Añadir título y etiquetas
            plt.title('Error Relativo (Absoluto) de Predicción por XGBoost')
            plt.xlabel('Carga del Motor Especificada [%] (rlsol_w)')
            plt.ylabel('Velocidad del Motor (nmot_w)')

            # Heatmap gris
            sns.heatmap(
                heatmap_reg_rel, 
                cmap=['lightgray'],
                cbar=False, 
                annot=annot_combinado_xgb, 
                fmt='', 
                linewidths=0.5,
                mask=~gray_mask,  # Invertir la máscara para aplicar el gris solo a las celdas fuera del rango
            )
            plt.gca().invert_yaxis() # Poner el origen del eje Y abajo

            # Ajustar las etiquetas del eje X a la izquierda
            plt.xticks(
                ticks=np.arange(len(heatmap_xgb_rel.columns)) + 0,
                labels=heatmap_xgb_rel.columns,
                rotation=0,
            )

            # Ajustar las etiquetas del eje y abajo
            plt.yticks(
                ticks=np.arange(len(heatmap_xgb_rel.index)) + 0,
                labels=heatmap_xgb_rel.index,
                rotation=0,
            )

            # Mostrar el gráfico
            st.pyplot(plt)
        st.success("Entrenamiento completado")