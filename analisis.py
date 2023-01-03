import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn import tree

from Generadorpdf import *
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from PIL import Image

def getDataFrame(archivo):
    if '.csv' in archivo:
        dataframe = pd.read_csv('/home/eduardo/Downloads/AnalisisDeDatos/archivos/'+archivo)
        return dataframe

    if '.xls' in archivo:
        dataframe = pd.read_excel('/home/eduardo/Downloads/AnalisisDeDatos/archivos/'+archivo)
        return dataframe

    if '.xlsx' in archivo:
        dataframe = pd.read_excel('/home/eduardo/Downloads/AnalisisDeDatos/archivos/'+archivo)
        return dataframe

    if '.json' in archivo:
        dataframe = pd.read_json('/home/eduardo/Downloads/AnalisisDeDatos/archivos/'+archivo)
        return dataframe
def generarGrafica(modelo, X, y, y_predict, titulo, etiqueta,  etiquetaX, etiquetaY, nombreImagen):
    import os
    import io
    dir = '/home/eduardo/Downloads/AnalisisDeDatos/imagenes/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    X_grid=np.arange(min(X),max(X),0.1)
    X_grid=X_grid.reshape((len(X_grid),1))
    plot.scatter(X,y,label='Datos reales', color='red')
    plot.plot(X,y_predict,label='Modelo',color='blue')
    plot.title(titulo)
    plot.xlabel(etiquetaX)
    plot.ylabel(etiquetaY)
    plot.savefig('/home/eduardo/Downloads/AnalisisDeDatos/imagenes/'+nombreImagen)
    plot.close()

    from base64 import encodebytes
    scriptDir = os.path.dirname(__file__)
    pil_img = Image.open(os.path.join(scriptDir,'/home/eduardo/Downloads/AnalisisDeDatos/imagenes/'+nombreImagen) , mode='r')
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    return encoded_img
def polinomial(x_name, y_name, archivoAnalisis, grados, titulo):
    now = datetime.now()
    try :
        dataframe = getDataFrame(archivoAnalisis)
        x = dataframe[x_name].values.reshape(-1, 1)
        y = dataframe[y_name].values.reshape(-1, 1)

        print('Informacion dataframe tratado')
        print(dataframe)

        modelo = PolynomialFeatures(degree=grados, include_bias=False)
        X_poly = modelo.fit_transform(x)

        lin_reg2 = LinearRegression()
        lin_reg2.fit(X_poly,y)
        y_entrenamiento = lin_reg2.predict(X_poly) #print(model.predict([[1.40],[1.90]]))

        mse = mean_squared_error(y,y_entrenamiento)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_entrenamiento)

        ecuacion ="Y(x) = "

        nombrePDF = now.strftime("%d%m%Y%H%M%S") + '.pdf'
        nombrePNG = now.strftime("%d%m%Y%H%M%S") + '.png'
        generarPDF(nombrePDF,titulo, 'Regresión Polinomial', '')

        return {
            "coeficiente": r2,
            "r2" : r2,
            "rmse" : rmse,
            "mse" : mse,
            #"predicciones" : valorpredicciones,
            "timestamp": now.strftime("%d/%m/%Y %H:%M:%S"),
            "code" : 200,
            "img" : generarGrafica(modelo, x, y, y_entrenamiento, titulo,  ecuacion, x_name , y_name, nombrePNG),
            "nombrePdf":nombrePDF,
            "archivo": generarPDF(nombrePDF, titulo, 'Regresión Polinomial', '')
        }
    except Exception as e:
        print('ERROR!!!!!!!!!!',str(e))
        return {
            "mensaje" : str(e).replace("\"", "-"),
            "code" : 666,
            "timestamp": now.strftime("%d/%m/%Y %H:%M:%S")
        }

def arbol(x_name, y_name, archivoAnalisis, titulo):
    df = pd.read_csv("/home/eduardo/Downloads/AnalisisDeDatos/archivos/penguins.csv").dropna().drop(columns="Unnamed: 0")
    x = df.drop(['species'], axis=1)
    y = df['species']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    x_test.reset_index(drop=True)
    y_test.reset_index(drop=True)

    x_train.reset_index(drop=True)
    y_train.reset_index(drop=True)

    x_train_tra = tratamento(x_train)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train_tra, y_train)

    fig = plot.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf,
                       feature_names=clf.feature_names_in_,
                       class_names=clf.classes_,
                       filled=True)
    plot.show()
    
def tratamento(df):
    encoder = ce.OneHotEncoder(cols='island', use_cat_names=True)
    df = encoder.fit_transform(df)
    df['sex'] = np.where(df['sex'] == 'FEMALE', 1, 0)
    df = df.fillna(0)
    return df

def redesBien(x_name, y_name, archivoAnalisis, titulo):
    df = pd.read_csv('/home/eduardo/Downloads/AnalisisDeDatos/archivos/diabetes.csv')
    target_column = ['Outcome']
    predictors = list(set(list(df.columns)) - set(target_column))
    df[predictors] = df[predictors] / df[predictors].max()
    df.describe().transpose()

    X = df[predictors].values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train, y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))

def gaussiano(x_name, y_name, archivoAnalisis, titulo):
    return {}



