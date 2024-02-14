# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:33:23 2024

@author: KGP
"""

# Redes Neuronales Artificiales


# Parte 1 - Pre procesado de datos (es un problema de clasificación porque intentamos predecir mediante binario si el cliente se queda o se va)

# Cómo importar las librerías
import numpy as np # contiene las herrarmientas matemáticas para hacer los algoritmos de machine learning
import matplotlib.pyplot as plt #pyplot es la sublibrería enfocada a los gráficos, dibujos
import pandas as pd #librería para la carga de datos, manipular, etc

# Importar el dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #cogemos de la columna 3 a la 12 (hay que decir +1 la última columna). Si quisieras columnas dispersas pondríamos entre []
y = dataset.iloc[:, 13].values
# iloc sirve para localizar por posición las variables, en este caso independientes
# hemos indicado entre los cochetes, coge todas las filas [:(todas las filas), :-1(todas las columnas excepto la última]
# .values significa que quiero sacar solo los valores del dataframe no las posiciones

# Codificar datos categóricos (en este caso las columnas 1 y 2)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Necesitamos una librería que permita traducir cada categoría en un número, le da igual que el dato sea categórico u ordinal 
from sklearn.compose import ColumnTransformer 
le_x_1 = preprocessing.LabelEncoder()
X[:, 1] = le_x_1.fit_transform(X[:, 1]) # Los países pasan a asignarseles un número
le_x_2 = preprocessing.LabelEncoder()
X[:, 2] = le_x_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer([("Geography", OneHotEncoder(), [1])], # The column numbers to be transformed (here is [1])
                                  remainder="passthrough") # Leave the rest of the columns untouched
X = onehotencoder.fit_transform(X)
X = X[:, 1:]
    # le_y = preprocessing.LabelEncoder(). En este caso estas dos líneas no hacen falta ya que la variable dependiente "y" ya biene en 0 y 1
    # y = le_y.fit_transform(y) En este caso estas dos líneas no hacen falta ya que la variable dependiente "y" ya biene en 0 y 1

# en este caso traducir les paises en un número no es del todo correcto, ya que no son categorías ordinales, uno no es mayor que el otro, como podría ser un conjunto de tallas de ropa
# *, aquí entran las variables dummy = one hot encoder, es una forma de traducir categorías que no siguen un orden, no son variables ordinales.
# si tengo tres etiquetas de paises, éstas pasan a convertirse en columnas y cada vez que una es asignada se le coloca un 1 y a las otras un 0 en la misma fila


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random_state podría coger cualquier número, es el número para poder reproducir el algoritmo

# Escalado de variables. Siguiente código COMENTADO porque se usa mucho pero no siempre. Si no hacemos este paso causaremos confusión en la red neuronal
from sklearn.preprocessing import StandardScaler # Utilizarlo para saber que valores debe escalar apropiadamente y luego hacer el cambio
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) #en el escalado de variables cuando tomo el conjunto de entrenamiento hago un fit transform para calcular el cambio de escala y aplicarlo a la matriz x_train.
x_test = sc_x.transform(x_test) #cuando hago el x_test solo hacemos transform sin "fit" de modo que la transformación se haga con los datos del transform de x_train



# Parte 2 - Construir la RNA

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential # Sequential sirve para inicializar los parámetros de la RN
from keras.layers import Dense # Dense crea las capas intermedias de la que consta la RN. Dense seria como la zona de sinapsis (no crea exactamente la capa), la conexión entre capas, tendré que especificar el tamaño de entrada y el de salida. Prepara los datos para la capa oculta que viene a continuación

# Inicianizar la RNA. Existen dos formas de hacerlo: Definir la secuencia de capas (escogido en este caso) o bien definir el grafo de como se van a relacionar entre sí las capas
classifier = Sequential()

# Añadir capas de entrada y primera capa oculta de la RN. Solo tenemos que definir cada una por separada e ir añadiéndolas a nuestro clasificador
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11)) # la capa Dense tendré que decirle que cualquier información que me entre tendré sea transformada a un espacio vectorial de dimensión que yo desee, en particular
        #el número de unidades ("units=") son cuantos nodos va a tener la capa oculta (sabemos los nodos de la capa de entrada, en este caso 11 variables indep. pero no de la oculta); una regla de oro (no formal) para calcular el número de nodos es que en la capa oculta se pone la media entre la cantidad de nodos de la capa de entrada y la de salida, en este caso sería (11+1)/2 = 6 (aunque al final podemos experimentar con otras cantidades)
        #los pesos los podemos inicializar de muchas maneras, en este caso cogemos kernel_initializer
        #activation es el tipo de activación, en este caso cogemos la función de rectificación lineal unitario para que cualquier valor positivo (aunque esté lejos de 1) se active y 0 o negativo se inactive

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu")) 

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid")) #cambiamos la función activación por una sigmoide porque necesitamos para la salida una de probabilidad (si cogieramos función escala iríamos directos a decir si es 1 o 0 sin saber la probabilidad)

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) #optimizador, el algoritmo para encontrar los pesos óptimos (anteriormente hemos puesto valores random)
        # loss es función que minimiza el error de pérdida
        
# Ajustamos la RNA al conjunto de Entrenamiento. Es el momento en el que la RNA aprende.
classifier.fit(x_train, y_train, batch_size = 10, epochs = 50)



# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5) #como tenemos datos de probabilidad establecemos un umbral para que los valores superiores a 0.5 (en este caso, podría ser cualquier otro) sean 1.
# Elaborar una matriz de confusión. Mide que tan bien ha evaluado el algoritmo la clasificación de los usuarios (en este caso) en compra o no compra con respecto a cómo estaban etiquetados
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)