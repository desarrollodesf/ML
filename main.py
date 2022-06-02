import streamlit as st
from joblib import load
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from os import listdir
from os.path import isfile, join
from PIL import Image

import os
import imghdr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal
import earthpy as et
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

import rasterio as rio
from features_from_band_colums_arrays import SpectralFeatures
from features_from_square_images_arrays_PRODUCCION import SpectralFeaturesSQ

fileOneSelection = None

# Configuración básica de la página
st.set_page_config(
  page_title="Plan de ordenamiento territorial", layout="wide",
)

# Clases proporcionadas por el negocio
classes = {
    2: "Ground",
    5: "Trees",
    6: "Buildings",
    9: "Water",
    17: "Bridge / elevated road",
    65: "Unlabeled"
}

# Al tener que realizar la conexión una vez, utilizamos el decorador para no hacer varios llamados a la vez
@st.experimental_singleton
def upload():
    models = {
    'clfAdaBoost':{
        "name":"Ada Boost",
        "model": None
        },
    'clfNearest Neighbors':{
        "name":"K-Nearest Neighbors",
        "model": None
        },
    'clfLogistic Regression':{
        "name":"Logistic Regression",
        "model": None
        },
    'clfNeural Net':{
        "name":"Neural Net",
        "model": None
        },
    'clfRandom Forest':{
        "name":"Random Forest",
        "model": None
        }
    }
    for name, values in models.items():
        values["model"] = load(f"classifiers/{name}.joblib")
    return models

models = upload() # Cargamos nuestros modelos
selection = st.selectbox("Models",models.keys(), index=0, format_func=lambda x: models.get(x)['name']) # Generamos una selección múltiple de nuestros modelos

def uploadImageNames():
    files = [f for f in listdir('images') if isfile(join('images', f))]
    return files

files = uploadImageNames()

fileOneSelection = st.selectbox("Imagen 1", files, 0)



fileTwoSelection = st.selectbox("Imagen 2", files , 1)


mostrarImagenes = st.button("Mostrar Imagenes seleccionadas") # Botón para predecir
if mostrarImagenes:
    
    image1 = Image.open('images/' + fileOneSelection)
    st.image(image1, caption=fileOneSelection, width = 500)   

    image2 = Image.open('images/' + fileTwoSelection)
    st.image(image2, caption=fileTwoSelection, width = 500)   
    

def predict(txt,selection):
    """
    Realiza la predicción del texto según los modelos seleccionados.
    Args:
        - txt. Texto a analizar.
        - selection. Selección de modelos a utilizar.
    """

    # Realizamos una gráfica mostrando los porcentajes de confianza de los modelos
    fig = make_subplots()
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

    for m in selection:
        p = models[m]["model"].predict([txt])[0] # Obtenemos la clase del modelo según la predicción
        proba = models[m]["model"].predict_proba([txt])[0] # Obtenemos las probabiidades de pertenencia de cada clase
        st.markdown(f'For **{models[m]["name"]}**, the class predicted is *{classes[p]}*') # Mostramos el resultado apra cada uno de los modelos seleccionados

        # Añadimos cada predicción según su probabilidad.
        fig.add_trace(
            go.Bar(
                x=list(classes.values()),
                y=proba,
                name=models[m]["name"]
            ))

    fig.update_yaxes(range=[0,1])
    fig.update_xaxes(title_text="Classes")
    st.plotly_chart(fig,use_container_width=True)


def create_array_images(path1,
                        end_multispectral_name="MSI.tif"):
    dataset = {'filename': [], 'MSI': []}
    #for file in os.listdir(path1 + "\\"):
    file = 'images/' + path1
    #if not os.path.isdir(path1 + file):  # Exclude folders
    type_file = imghdr.what(file)  # type of image
    supported_images = ["gif", "jpg", "png", "tiff", "bmd"]
    image_is_accepted = any(type_file in supported_images
                            for _ in
                            supported_images)  # Check if the image ir supported
    if image_is_accepted:
        the_array_image = gdal.Open('images/' + path1, gdal.GA_ReadOnly).ReadAsArray()
        if file.endswith(end_multispectral_name):
            dataset['MSI'].append(the_array_image)
            dataset['filename'].append(file[0:(len(file) - len(end_multispectral_name) - 1)])
    return pd.DataFrame(dataset)

def create_filter_column(data, percentage):
    # Select random cells and write the indices row, cols, stacked
    # to an additional column of data
    data["RND_IDX"] = np.nan
    data["RND_IDX"] = data["RND_IDX"].astype(object)

    for row_index in range(len(data)):
        # the_array = self.raw_data["AGL"].iloc[row_index]
        bands_rows_cols = data['MSI'].iloc[row_index].shape
        x_rnd_indexes = np.random.permutation(np.arange(bands_rows_cols[1]))[0:round(bands_rows_cols[1] * percentage)]
        y_rnd_indexes = np.random.permutation(np.arange(bands_rows_cols[2]))[0:round(bands_rows_cols[2] * percentage)]
        the_indexes = {}
        the_indexes["xidx"] = []
        the_indexes["yidx"] = []
        # for (idx, column_name) in enumerate(data.columns):
        # if idx != 0 and idx != (len(data.columns)-1):
        the_indexes["vidx"] = []
        bands_rows_cols = (data["MSI"].iloc[row_index]).shape

        the_shape = (data["MSI"].iloc[row_index]).shape
        if len(the_shape) > 2:
            # number_of_bands = the_shape[0]
            # nrows = the_shape[1]
            ncols = the_shape[2]
        else:
            # number_of_bands = 1
            # nrows = the_shape[0]
            ncols = the_shape[1]
            # for bands in range(number_of_bands):
        for (index, ind_x) in enumerate(x_rnd_indexes):
            ind_y = y_rnd_indexes[index]
            # for ind_y in y_rnd_indexes:
            # if idx == 1:
            the_indexes["xidx"].append(ind_x)
            the_indexes["yidx"].append(ind_y)
            # vi = bands*nrows*ncols + (ind_x * ncols) + (ind_y + 1 )
            vi = (ind_x * ncols) + (ind_y + 1)
            the_indexes["vidx"].append(vi)
        data.at[row_index, "RND_IDX"] = pd.DataFrame(the_indexes)   

def stack_data(raw_data):
    # Input data has multiple bands
    number_of_columns = 0
    for (idx, column) in enumerate(raw_data.columns):
        if (column != "filename") and (column != "RND_IDX") and (column != "CLS"):
            bands_cols_rows = (raw_data[column].iloc[0]).shape
            if len(bands_cols_rows) > 2:
                number_of_columns += bands_cols_rows[0]
            else:
                number_of_columns += 1

    # Output data has a column for each band in input data
    predictors = np.full(shape=(0, number_of_columns), fill_value=None, dtype=None)
    targets = np.full(shape=(0, 1), fill_value=None, dtype=int)
    for row_index in range(len(raw_data)):
        vidx = ((raw_data["RND_IDX"].iloc[row_index])["vidx"]).tolist()
        # vidx = df_filters["vidx"]
        # import numpy as np
        # 1) bands(0), rows(1), cols(2):
        # arr = np.arange(16).reshape((2, 2, 4))
        # first row, both images (horizontal results)
        # arr[:][0]
        # 2) rows(0) (before 1), cols(1) (before 2), bands(2) (before 0) (TRANSPOSED RESULTS!)
        # arr2 = arr.transpose(1,2,0) # Pass old indexes to the the transpose
        # in the new order rows, cols, bands.
        # first row, both images (vertical results):
        # arr2[0]
        # first row, first column, both images (flat results)
        # arr2[0][0]
        # 3) leave only one column for each image
        # arr3 = arr2.reshape(-1,arr.shape[0])
        # note: reshape -1 remove all the dimensions, then, split the array in the number of bans: arr.shape[0]
        #
        # read all pixel values of 8 bands (MSI) (one column for each band)
        #
        array_msi = raw_data['MSI'].iloc[row_index]  # bands, rows, cols
        array_msi = array_msi.transpose(1, 2, 0).reshape(-1, array_msi.shape[0])  # new cols are the number of bands
        array_msi = np.take(array_msi, vidx, axis=0)

        #
        # PC
        array_pc = raw_data['PC'].iloc[row_index]
        pc_bands_rows_cols = array_pc.shape
        if len(pc_bands_rows_cols) > 2:
            pc_n_bands = pc_bands_rows_cols[0]
            array_pc = array_pc.transpose(1, 2, 0).reshape(-1, pc_n_bands)
        else:
            pc_n_bands = 1
            array_pc = array_pc.reshape(-1, 1)
        array_pc = np.take(array_pc, vidx, axis=0)

        #
        # ICA
        array_ica = raw_data['ICA'].iloc[row_index]
        ica_bands_rows_cols = array_ica.shape
        if len(ica_bands_rows_cols) > 2:
            ica_n_bands = ica_bands_rows_cols[0]
            array_ica = array_ica.transpose(1, 2, 0).reshape(-1, ica_n_bands)
        else:
            ica_n_bands = 1
            array_ica = array_ica.reshape(-1, 1)
        array_ica = np.take(array_ica, vidx, axis=0)

        
        predictors_names = ['B', 'G','R']
        #array_mp = np.take(array_mp, vidx, axis=0)

        for band in range(pc_n_bands):
            predictors_names.append("PC" + str(band + 1))
        for band in range(ica_n_bands):
            predictors_names.append("ICA" + str(band + 1))
        # for band in range(mp_n_bands):
        # predictors_names.append("MORPHO" + str(band+1))
        predictors = np.concatenate((predictors, np.concatenate((array_msi,
                                                                 array_pc, array_ica), axis=1)), axis=0)
        #targets = np.concatenate((targets), axis=0)
    targets = targets.reshape(-1)
    return predictors, targets, predictors_names

def processImage(imagePath, classifierPath):
    path_testing = os.getcwd() + "\\images\\"
    data_testing = create_array_images(imagePath)

    my_spectral_featuresSQ = SpectralFeaturesSQ(data_testing)
    my_spectral_featuresSQ.pca()
    my_spectral_featuresSQ.ica()
    create_filter_column(data_testing, percentage=1)

    predictors_testing, targets_testing, predictors_testing_names = stack_data(data_testing)
    predictors_testing_pd = pd.DataFrame(predictors_testing, columns=predictors_testing_names)
    data = predictors_testing_pd[['B','G','R']]

    # return data
    loaded_model = joblib.load('classifiers/' + classifierPath + '.joblib')
    predictions = loaded_model.predict(data)

    return predictions

pred = st.button("Predict") # Botón para predecir
if pred:
    resultado1 = processImage(fileOneSelection, selection)
    resultado2 = processImage(fileTwoSelection, selection)

    txt = st.text_area("Diccionario de Clase", value= str(classes), height = 30)


    (unique1, counts1) = np.unique(resultado1, return_counts=True)
    frequencies1 = np.asarray((unique1, counts1)).T
    classes1 = frequencies1[ :,0]
    dfArrayFrecuency1 = pd.DataFrame(frequencies1) 
    dfArrayFrecuency1 = dfArrayFrecuency1.rename({ 1:"Valor"}, axis=1)
    dfArrayFrecuency1 = dfArrayFrecuency1.rename({ 0:"Clase"}, axis=1)
    plotFrecuency1 = dfArrayFrecuency1.drop(labels=["Clase"], axis=1)
    txt1 = st.text_area("Resulto predicción imagen 1", value= str(dfArrayFrecuency1), height = 140)
    st.bar_chart(data=plotFrecuency1)

    (unique2, counts2) = np.unique(resultado2, return_counts=True)
    frequencies2 = np.asarray((unique2, counts2)).T
    classes2  = frequencies2[ :,0]
    dfArrayFrecuency2 = pd.DataFrame(frequencies2) 
    dfArrayFrecuency2 = dfArrayFrecuency2.rename({ 1:"Valor"}, axis=1)
    dfArrayFrecuency2 = dfArrayFrecuency2.rename({ 0:"Clase"}, axis=1)
    plotFrecuency2 = dfArrayFrecuency2.drop(labels=["Clase"], axis=1)
    txt2 = st.text_area("Resulto predicción imagen 2", value= str(dfArrayFrecuency2), height = 140)
    st.bar_chart(data=plotFrecuency2)

    classesTotal = np.concatenate((classes1, classes2), axis=0, out=None, dtype=None, casting="same_kind")
    uniqueClasses = np.unique(classesTotal, return_index=False, return_inverse=False, return_counts=False, axis=None)
    
    dfMetrica = pd.DataFrame(frequencies2) 
    

    #txt1 = st.text_area("Resulto predicción imagen 1", value= str(dfMetrica.loc[dfMetrica[0]==5][1].values[0] ), height = 140)

    #k = 0
    #while k < uniqueClasses.size:
    #    col1, col2, col3= st.columns(3)
    #    if dfMetrica.loc[dfMetrica[0]==uniqueClasses[k]] is not None:
    #        col1.metric(classes[uniqueClasses[k]], dfMetrica.loc[dfMetrica[0]==uniqueClasses[k]][1].values[0], "1.2 °F")
    #        k += 1