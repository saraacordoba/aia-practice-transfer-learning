#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2025-26
# PARTE I del trabajo práctico: Implementación de árboles de decisión 
#                               y random forests
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================



import math
import random
import numpy as np



# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar (casi) todos los conjuntos de datos,
# basta con tener descomprimido el archivo datos-trabajo-aia.zip (en el mismo sitio
# que este archivo) Y CARGARLOS CON LA SIGUIENTE ORDEN:
    
from carga_datos import *    

# Como consecuencia de la línea anterior, se habrán cargado los siguientes 
# conjuntos de datos, que pasamos a describir, junto con los nombres de las 
# variables donde se cargan. Todos son arrays de numpy: 


# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre pasajeros del Titanic y si sobrevivieron o no. Es una versión 
#   restringida de este conocido dataset, con solo tres caracteristicas:
#   Pclass, IsFemale y Age. Se carga en las variables X_train_titanic, 
#   y_train_titanic, X_test_titanic e y_test_titanic.

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (0:republicano o 1:demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos (ver 
#   descripción en votos.py)


# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer. 
#   Ver descripcición en sikit learn.

  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos del dataset original. 
#   Los textos se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    


#  Además, en la carpeta datos/ se tienen los siguientes datasets, que
#  habrán de ser procesado y cargado (es decir, no se caragan directamente con
#  carga_datos.py).   
    
# * Un archivo credito.csv con datos sobre concesión de prestamos en una entidad 
#   bancaria, en función de: tipo de empleo, si ya tiene productos finacieros 
#   contratados, número de propiedades, número de hijos, estado civil y nivel de 
#   ingresos (cargarlo usando pd.read_csv en arrays de numpy X_credito e y_credito,
#   donde X_credito son las seis primeras columnas e y_credito la última).


# * Un archivo adultDataset.csv, con datos de personas para poder predecir si
#   alguien gana más o menos de 50000 dólares anuales, en función de una serie 
#   de características (para más detalles, ver https://archive.ics.uci.edu/dataset/2/adult)  
#   Más adelante se explica cómo cargar y procesar este conjunto de datos. 

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En la carpeta digitdata están todos los datos en archivos de texto. 
#   Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 




# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

#  >>>Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#   (array(['democrata', 'republicano'], dtype='<U11'), array([267, 168]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array(['democrata', 'republicano'], dtype='<U11'), array([178, 112]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array(['democrata', 'republicano'], dtype='<U11'), array([89, 56]))

# La división en trozos es aleatoria y en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
#  (array([0, 1]), array([136, 229]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([34, 57]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------



def particion_entr_prueba(X, y, test=0.20):
    indices_entrenamiento = []
    indices_prueba = []

    clases = np.unique(y)
    
    for c in clases:
        indices_clase = np.where(y == c)[0] # el 0 porque devuelve una tupla

        indices_clase_mezclados = np.random.permutation(indices_clase)
        
        n_test = int(np.round(len(indices_clase) * test))
        
        indices_clase_test = indices_clase_mezclados[:n_test]
        indices_clase_train = indices_clase_mezclados[n_test:]
        
        indices_prueba.extend(indices_clase_test)
        indices_entrenamiento.extend(indices_clase_train)
        
    indices_entrenamiento.sort()
    indices_prueba.sort()
    
    X_train = X[indices_entrenamiento]
    X_test = X[indices_prueba]
    y_train = y[indices_entrenamiento]
    y_test = y[indices_prueba]

    return X_train, X_test, y_train, y_test






# ===============================================
# EJERCICIO 2: IMPLEMENTACIÓN ÁRBOLES DE DECISIÓN
# ===============================================


# En este ejercicio pedimos implementar en python un algoritmo de aprendizaje para árboles 
# de decisión. Los árboles de decisión que trataremos serán árboles binarios, en los que
# en cada nodo interior se pregunta por el valor de un atributo o característica dada, 
# y si ese valor es mayor o menor que un valor umbral dado. Este es el mismo tipo de árbol 
# de decisión que se  manejan en Scikit Learn. 

# Se puede obtener información de este tipo de árboles en la entrada "Decision Trees"
# del manual de Scikit Learn. También en la práctica del Titanic hecha en clase.

# Se propone la implementación de un clasificador basado en árboles de
# de decisión, entrenado usando el algoritmo CART, similar al que implementa 
# la clase DecisonTree de Scikit Learn, pero con ALGUNAS VARIANTES, que indicaremos más
# adelante.

# Los árboles de decisión están formados por nodos. Usar la siguiente clase para la
# implementación de los nodos:
    
class Nodo:
    def __init__(self, atributo=None, umbral=None, izq=None, der=None,distr=None,*,clase=None):
        self.atributo = atributo
        self.umbral = umbral
        self.izq = izq
        self.der = der
        self.distr= distr
        self.clase = clase
        
    def es_hoja(self):
        return self.clase is not None

# Pasamos a describir los distintos atributos de esta clase:

# - atributo: el atributo por el que se pregunta en el nodo. Referenciaremos a cada
#   atributo POR EL ÍNDICE DE SU POSICIÓN (el número de columna).
# - umbral: es el valor umbral por el que se pregunta en el nodo. Si la instancia tiene un
#   valor de atributo menor o igual que el umbral, se sigue por el subárbol izquierdo. En
#   caso contrario, por el subárbol derecho.
# - izq: es el nodo raiz del subárbol izquierdo.
# - der: el nodo raiz del subárbol derecho.
# - distr: es un diccionario cuyas claves son las posibles clases, y cuyos valores son
#   cuántos ejemplos del conjunto de entrenamiento correspondientes al nodo hay de cada
#   clase. Cuando decimos "ejemplos correspondientes al nodo" queremos decir aquellos que
#   cumplen todas las condiciones (desde la raiz) que llevan a ese nodo.
# - clase: Si el nodo es una hoja, es la clase que predice. Si no es una hoja, este valor es None.



# Lo que sigue es una descripción del algoritmo que se pide implementar para la
# construcción de un árbol de decisión. En principio describiremos la versión básica y más
# conocida, y posteriormente indicaremos las peculiaridades y variantes que pedimos
# introducir a esta versión básica.

# Supondremos que recibimos un conjunto de entrenamiento X,y y además dos valores max_prof
# y min_ejemplos_nodo_interior, que nos van a servir como condiciones adicionales para
# dejar de expandir un nodo. El algoritmo se define recursivamente y tiene además un
# argumento adicional prof (inicialmente 0), con la profundidad del nodo actual.  

# CONSTRUYE_ARBOL(X,y,min_ejemplos_nodo_interior,max_prof,prof=0):

# 1. SI prof es mayor o igual que max_prof, 
#       o el número de ejemplos de X es menor que min_ejemplos_nodo_interior,
#       o en X todos los ejemplos son de la misma clase:
#       ENTONCES:
#          Devolver un nodo hoja con la distribución de clases en X,
#                    y con la clase mayoritaria en X
# 2. EN OTRO CASO:
#        encontrar el MEJOR atributo A y el mejor umbral u para ese atributo
#        y particionar en dos tanto X como y:
#            * X_izq, y_izq los ejemplos cuyo valor de A es menor o igual que u
#            * X_der, y_der los ejemplos cuyo valor de A es mayor que u
#        Llamadas recursivas:
#            A_izq=CONSTRUYE_ARBOL(X_izq,y_izq,min_ejemplos_nodo_interior,max_prof,prof+1)
#            A_der=CONSTRUYE_ARBOL(X_der,y_der,min_ejemplos_nodo_interior,max_prof,prof+1)
#        Devolver un nodo interior con el atributo y umbral seleccionado,
#                 con la distribución de clases de X, y con A_izq y A_der
#                 como hijos izquierdo y derecho respectivamente.


# Lo anterior es la descripción básica. A continuación indicamos una serie de variantes y
# cuestiones adicionales que se le piden a esta implementación concreta:

# - Consideraremos la posibilidad de restringir los atributos a usar en el árbol a un
#   número de atributos dado n_atrs. Ese subconjunto de atributos se seleccionará
#   aleatoriamente al principio de la construcción del aŕbol y será el mismo para todos
#   los nodos.
#   Por ejemplo, si el dataset tiene 15 atributos y le damos n_atrs=9, al comienzo de la
#   construcción del árbol seleccionamos aleatoriamente 9 atributos, y ya en los nodos del
#   árbol solo podrán aparecer alguno de esos 9 atributos. Nótese que si n_atrs es igual
#   al total de atributos, tendríamos la versión estándar del algoritmo.
#   NOTA: téngase en cuenta que a diferencia de lo que ocurre en la versión clásica de
#   Random Forests, no sorteamos los atributos en cada nodo, sino que hay un único sorteo
#   inicial para todo el árbol.

# - A la hora de elegir el mejor atributo y umbral para la partición de los nodos
#   interiores, usar el criterio de mejor GANANCIA DE INFORMACIÓN (en particular, NO USAR GINI).

# - La principal carga computacional de este algoritmo se debe a la cantidad de candidatos a
#   mejor atributo y mejor umbral que hay que evaluar en cada nodo, para decidir cuál es
#   la mejor partición. El hecho de limitar el número de atributos candidatos (como se ha
#   descrito más arriba), va en esa dirección. 
#   Otra manera es limitar también los posibles valores umbrales a considerar
#   para cada atributo. Para ello, en la implementación que se pide actuaremos en dos
#   sentidos:
#      (a) Considerar solo como candidatos a umbral los puntos medios entre cada par de 
#         valores consecutivos del atributo en los que hay cambio de clase, para los
#         ejemplos correspondientes a ese nodo.
#         Por ejemplo, si ordenados los valores del atributo A en orden creciente, hay un
#         ejemplo con valor v1 de A y clase C1 y a continuación otro ejemplo con valor v2
#         en A y clase C2 distinta de C1, entonces (v1+v2)/2 es un posible valor umbral
#         candidato. El resto de valores NO se considera candidato.

#      (b) En cada nodo, para elegir los umbrales candidatos correspondientes a un atibuto,
#         no considerar todos los ejemplos que corresponden a ese nodo, sino 
#         sólo  una proporción de los mismos, seleccionada aleatoriamente. La proporción a
#         considerar se da en un parámetro prop_umbral.
#         Por ejemplo, si prop_umbral es 0.7 y el conjunto de ejemplos correspondientes al
#         nodo es de 200 ejemplos, entonces aplicaremos el proceso de selección de
#         umbrales candidatos descrito en (a) considerando sólo un suconjunto de 140
#         ejemplos seleccionado aleatoriamente de entre esos 200.  



# Con las descripciones anteriores, ya podemos precisar lo que se pide en este apartado. 
# Se pide implementar una clase ArbolDecision con el siguiente formato:
  

# class ArbolDecision:
#     def __init__(self, min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
#         ......
#                
#     def entrena(self, X, y):
#         .......
#        
#     def clasifica(self, X):
#         .......
#
#     def clasifica_prob(self, x):
#         .......
#
#     def imprime_arbol(self,nombre_atrs,nombre_clase) :
#         .......



#  El constructor tiene los siguientes argumentos de entrada:

#     + min_ejemplos_nodo_interior: mínimo número de ejemplos del conjunto de 
#       entrenamiento en un nodo del árbol que se aprende, para que se considere 
#       su división.  
#     + max_prof: profundidad máxima del árbol que se aprende.
#     + n_atrs: número de atributos candidatos a considerar en cada partición
#     + prop_umbral: proporción de ejemplos a considerar cuando se buscan los 
#       umbrales candidatos.    
  
#      

# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente.
#     

# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Método clasifica_prob: recibe UN EJEMPLO y devuelve un diccionario con la predicción
#   de probabilidad de pertenecer a cada clase. Esa probabilidad se calcula como la
#   proporción de ejemplos de clase en la distribución del nodo hoja que da la
#   predicción.

# * Método imprime_arbol: recibe la lista de nombres de cada atributo (columnas) y el
#   nombre del atributo de clasificación, e imprime el árbol de decisión aprendido 
#   (ver ejemplos más abajo) [SUGERENCIA: hacerlo con una función auxiliar recursiva] 


# Si se llama al método de clasificación, o al de impresión, antes de entrenar el modelo,
# se debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        




# Algunos ejemplos (los resultados pueden variar, debido a la aleatoriedad)
# **************************************************************************

# TITANIC
# -------

# >>> clf_titanic = ArbolDecision(max_prof=3,min_ejemplos_nodo_interior=5,n_atrs=3)
# >>> clf_titanic.entrena(X_train_titanic, y_train_titanic)
# >>> clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"],"Sobrevive")

# Mujer <= 0.000
#      Edad <= 11.000
#           Pclass <= 2.500
#                Sobrevive: 1 -- {1: 10}
#           Pclass > 2.500
#                Sobrevive: 0 -- {0: 13, 1: 8}
#      Edad > 11.000
#           Pclass <= 1.000
#                Sobrevive: 0 -- {0: 62, 1: 30}
#           Pclass > 1.000
#                Sobrevive: 0 -- {0: 270, 1: 32}
# Mujer > 0.000
#      Pclass <= 2.000
#           Edad <= 2.000
#                Sobrevive: 0 -- {0: 1, 1: 1}
#           Edad > 2.000
#                Sobrevive: 1 -- {0: 5, 1: 122}
#      Pclass > 2.000
#           Edad <= 38.500
#                Sobrevive: 1 -- {0: 46, 1: 58}
#           Edad > 38.500
#                Sobrevive: 0 -- {0: 9, 1: 1}

# VOTOS
# -----

# >>> clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=16)
# >>> clf_votos.entrena(Xe_votos, ye_votos)
# >>> nombre_atrs_votos=[f"Votación {i}" for i in range(1,17)]
# >>> clf_votos.imprime_arbol(nombre_atrs_votos,"Partido")

# Votación 4 <= 0.000
#      Votación 3 <= 0.000
#           Votación 11 <= 0.000
#                Votación 13 <= 0.500
#                     Votación 14 <= -0.500
#                          Partido: democrata -- {'democrata': 2}
#                     Votación 14 > -0.500
#                          Partido: republicano -- {'republicano': 3}
#                Votación 13 > 0.500
#                     Votación 7 <= -1.000
#                          Partido: democrata -- {'democrata': 1, 'republicano': 1}
#                     Votación 7 > -1.000
#                          Partido: democrata -- {'democrata': 4}
#           Votación 11 > 0.000
#                Partido: democrata -- {'democrata': 11}
#      Votación 3 > 0.000
#           Partido: democrata -- {'democrata': 149}
# Votación 4 > 0.000
#      Votación 11 <= 0.500
#           Votación 10 <= -1.000
#                Votación 12 <= -1.000
#                     Votación 3 <= -1.000
#                          Partido: democrata -- {'democrata': 1, 'republicano': 1}
#                     Votación 3 > -1.000
#                          Partido: republicano -- {'republicano': 2}
#                Votación 12 > -1.000
#                     Votación 3 <= 0.000
#                          Partido: republicano -- {'republicano': 35}
#                     Votación 3 > 0.000
#                          Partido: republicano -- {'democrata': 1, 'republicano': 2}
#           Votación 10 > -1.000
#                Partido: republicano -- {'republicano': 55}
#      Votación 11 > 0.500
#           Votación 7 <= -1.000
#                Votación 3 <= -1.000
#                     Votación 13 <= 0.000
#                          Partido: democrata -- {'democrata': 1}
#                     Votación 13 > 0.000
#                          Partido: republicano -- {'democrata': 2, 'republicano': 9}
#                Votación 3 > -1.000
#                     Partido: democrata -- {'democrata': 6}
#           Votación 7 > -1.000
#                Partido: republicano -- {'republicano': 4}


# IRIS
# ----

    
# >>> clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
# >>> clf_iris.entrena(X_train_iris, y_train_iris)
# >>> clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")



#  Long. Pétalo <= 2.450
#       Clase: 0 -- {0: 33}
#  Long. Pétalo > 2.450
#       Long. Pétalo <= 4.900
#            Anch. Pétalo <= 1.650
#                 Clase: 1 -- {1: 32}
#            Anch. Pétalo > 1.650
#                 Clase: 2 -- {1: 1, 2: 3}
#       Long. Pétalo > 4.900
#            Clase: 2 -- {2: 30}


# CÁNCER DE MAMA
# --------------

# >>> clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# >>> clf_cancer.entrena(Xev_cancer, yev_cancer)

# >>> nombre_atrs_cancer=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#        'mean smoothness', 'mean compactness', 'mean concavity',
#        'mean concave points', 'mean symmetry', 'mean fractal dimension',
#        'radius error', 'texture error', 'perimeter error', 'area error',
#        'smoothness error', 'compactness error', 'concavity error',
#        'concave points error', 'symmetry error',
#        'fractal dimension error', 'worst radius', 'worst texture',
#        'worst perimeter', 'worst area', 'worst smoothness',
#        'worst compactness', 'worst concavity', 'worst concave points',
#        'worst symmetry', 'worst fractal dimension']

# >>> clf_cancer.imprime_arbol(nombre_atrs_cancer,"Es benigno")


#  mean concave points <= 0.051
#       mean area <= 696.050
#            area error <= 34.405
#                 mean area <= 505.550
#                      Es benigno: 1 -- {1: 172}
#                 mean area > 505.550
#                      worst texture <= 30.145
#                           mean concave points <= 0.050
#                                Es benigno: 1 -- {1: 63}
#                           mean concave points > 0.050
#                                Es benigno: 0 -- {0: 1, 1: 1}
#                      worst texture > 30.145
#                           mean texture <= 24.840
#                                compactness error <= 0.013
#                                     Es benigno: 0 -- {0: 3}
#                                compactness error > 0.013
#                                     Es benigno: 1 -- {1: 2}
#                           mean texture > 24.840
#                                Es benigno: 1 -- {1: 11}
#            area error > 34.405
#                 mean concave points <= 0.032
#                      Es benigno: 1 -- {1: 7}
#                 mean concave points > 0.032
#                      mean perimeter <= 89.175
#                           Es benigno: 0 -- {0: 3}
#                      mean perimeter > 89.175
#                           mean texture <= 20.115
#                                Es benigno: 1 -- {1: 3}
#                           mean texture > 20.115
#                                Es benigno: 0 -- {0: 1}
#       mean area > 696.050
#            mean texture <= 16.190
#                 Es benigno: 1 -- {1: 4}
#            mean texture > 16.190
#                 worst fractal dimension <= 0.066
#                      Es benigno: 1 -- {1: 2}
#                 worst fractal dimension > 0.066
#                      Es benigno: 0 -- {0: 6}
#  mean concave points > 0.051
#       mean area <= 790.850
#            worst texture <= 25.655
#                 mean concave points <= 0.079
#                      mean concave points <= 0.052
#                           Es benigno: 0 -- {0: 1}
#                      mean concave points > 0.052
#                           Es benigno: 1 -- {1: 20}
#                 mean concave points > 0.079
#                      Es benigno: 0 -- {0: 6}
#            worst texture > 25.655
#                 perimeter error <= 1.558
#                      Es benigno: 0 -- {0: 1, 1: 1}
#                 perimeter error > 1.558
#                      Es benigno: 0 -- {0: 37}
#       mean area > 790.850
#            Es benigno: 0 -- {0: 111}



# EJEMPLOS DE RENDIMIENTOS OBTENIDOS CON LOS CLASIFICADORES:
# ----------------------------------------------------------

# Usamos la siguiente función para medir el rendimiento (proporción de aciertos) 
# de un clasificador sobre un conjunto de ejemplos:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/X.shape[0]
    

# Ejemplos (obviamente, el resultado puede variar):


# >>> rendimiento(clf_titanic,X_train_titanic,y_train_titanic)
# 0.8158682634730539
# >>> rendimiento(clf_titanic,X_test_titanic,y_test_titanic)
# 0.7982062780269058

# >>> rendimiento(clf_votos,Xe_votos,ye_votos)
# 0.9827586206896551
# >>> rendimiento(clf_votos,Xp_votos,yp_votos)
# 0.9310344827586207

# >>> rendimiento(clf_iris,X_train_iris,y_train_iris)
#  0.98989898989899
# >>> rendimiento(clf_iris,X_test_iris,y_test_iris)
# 0.9607843137254902

# >>> rendimiento(clf_cancer,Xev_cancer,yev_cancer)
# 0.9956140350877193
# >>> rendimiento(clf_cancer,Xp_cancer,yp_cancer)
# 0.9557522123893806



class ArbolDecision:

    def __init__(self, min_ejemplos_nodo_interior=5, max_prof=10, n_atrs=10, prop_umbral=1.0):
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        self.raiz = None
        

    def entrena(self, X, y):
        total_atributos = X.shape[1]
        n_atrs_real = min(self.n_atrs, total_atributos)
        self.atributos_seleccionados = np.random.choice(total_atributos, n_atrs_real, replace=False) #replace = false pa q no de atributos repes
        self.raiz = self._construye_arbol(X, y, prof=0)


    def _construye_arbol(self, X, y, prof):
        num_ejemplos = X.shape[0]
        clases_unicas, conteos = np.unique(y, return_counts=True)
        
        distr_clases = dict(zip(clases_unicas, conteos))
        clase_mayoritaria = clases_unicas[np.argmax(conteos)]

        if (prof >= self.max_prof or 
            num_ejemplos < self.min_ejemplos_nodo_interior or 
            len(clases_unicas) == 1):
            
            return Nodo(distr=distr_clases, clase=clase_mayoritaria)




        mejor_ganancia = -1.0
        mejor_atrib = None
        mejor_umbral = None

        for atr in self.atributos_seleccionados:
            valores_atr = X[:, atr]

            #prop umbral es el numero de ejemplos q voy a mirar pa q no sea tan lento 
            if self.prop_umbral < 1.0:
                n_sub = int(np.round(num_ejemplos * self.prop_umbral))
                # Sorteamos un porcentaje de las filas para buscar los umbrales candidatos
                indices_sub = np.random.choice(num_ejemplos, max(1, n_sub), replace=False)
                valores_cand = valores_atr[indices_sub]
                y_cand = y[indices_sub]
            else:
                valores_cand = valores_atr
                y_cand = y



            indices_ordenados = np.argsort(valores_cand)
            vals_ord = valores_cand[indices_ordenados]
            y_ord = y_cand[indices_ordenados]
            
            umbrales_validos = []
            for i in range(len(vals_ord) - 1):
                # Si el vecino siguiente cambia de clase y no repite el mismo valor exacto
                if y_ord[i] != y_ord[i+1] and vals_ord[i] != vals_ord[i+1]:
                    punto_medio = (vals_ord[i] + vals_ord[i+1]) / 2.0
                    umbrales_validos.append(punto_medio)

            for umbral in umbrales_validos:
                # Simulamos cómo se dividirían los datos reales del nodo con este umbral
                indices_izq = np.where(valores_atr <= umbral)[0]
                indices_der = np.where(valores_atr > umbral)[0]
                
                ganancia = ganancia_informacion(y, y[indices_izq], y[indices_der])
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_atrib = atr
                    mejor_umbral = umbral

        if mejor_atrib is None:
            return Nodo(distr=distr_clases, clase=clase_mayoritaria)

        

        valores_ganadores = X[:, mejor_atrib]
        mask_izq = valores_ganadores <= mejor_umbral  
        mask_der = ~mask_izq                         
     
        hijo_izq = self._construye_arbol(X[mask_izq], y[mask_izq], prof + 1)
        hijo_der = self._construye_arbol(X[mask_der], y[mask_der], prof + 1)
    
        return Nodo(atributo=mejor_atrib, umbral=mejor_umbral, izq=hijo_izq, der=hijo_der, distr=distr_clases)
    


    def clasifica(self, X):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("Debes entrenar el árbol antes de clasificar.")
            
        predicciones = []
        for i in range(X.shape[0]):
            ejemplo = X[i, :]
            
            nodo_actual = self.raiz
            while not nodo_actual.es_hoja():
                valor = ejemplo[nodo_actual.atributo]
                if valor <= nodo_actual.umbral:
                    nodo_actual = nodo_actual.izq  
                else:
                    nodo_actual = nodo_actual.der  
            
            
            predicciones.append(nodo_actual.clase)
            
        return np.array(predicciones)
    


    def clasifica_prob(self, x):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("Debes entrenar el árbol antes de clasificar.")
        
        nodo_actual = self.raiz
        while not nodo_actual.es_hoja():
            valor = x[nodo_actual.atributo]
            if valor <= nodo_actual.umbral:
                nodo_actual = nodo_actual.izq
            else:
                nodo_actual = nodo_actual.der
        
        total_ejemplos_hoja = sum(nodo_actual.distr.values())
        probabilidades = {}
        for clase, conteo in nodo_actual.distr.items():
            probabilidades[clase] = conteo / total_ejemplos_hoja  
        return probabilidades



    def imprime_arbol(self, nombre_atrs, nombre_clase):
        if self.raiz is None:
            raise ClasificadorNoEntrenado("Debes entrenar el árbol antes de imprimirlo.")
            
        self._pintar_nodo(self.raiz, nombre_atrs, nombre_clase, prof=0)


    def _pintar_nodo(self, nodo, nombre_atrs, nombre_clase, prof):
        margen = "     " * prof
        
        
        if nodo.es_hoja():
            print(margen + str(nombre_clase) + ": " + str(nodo.clase) + " -- " + str(nodo.distr))
            
        
        else:
            nombre_del_atributo = nombre_atrs[nodo.atributo]
            
            umbral_texto = f"{nodo.umbral:.3f}"

            print(margen + str(nombre_del_atributo) + " <= " + umbral_texto)
            self._pintar_nodo(nodo.izq, nombre_atrs, nombre_clase, prof + 1)
            
            print(margen + str(nombre_del_atributo) + " > " + umbral_texto)
            self._pintar_nodo(nodo.der, nombre_atrs, nombre_clase, prof + 1)



def entropia(y):
    if len(y) == 0:
        return 0.0
        
    clases, conteos = np.unique(y, return_counts=True)
    
    probabilidades = conteos / len(y)
    
    return -np.sum(probabilidades * np.log2(probabilidades))


def ganancia_informacion(y_padre, y_izq, y_der):

    entropia_padre = entropia(y_padre)
    
    peso_izq = len(y_izq) / len(y_padre)
    peso_der = len(y_der) / len(y_padre)
    
    entropia_hijos = (peso_izq * entropia(y_izq)) + (peso_der * entropia(y_der))
    
    return entropia_padre - entropia_hijos






















# =============================================
# EJERCICIO 3: IMPLEMENTACIÓN DE RANDOM FORESTS
# =============================================

# Usando la clase ArbolDecision, implementar un clasificador Random Forest. 

# Un clasificador Random Forest aplica dos técnicas que reducen el sobreajuste que 
# pudiéramos tener con un único árbol de decisión:

# - En lugar de aprender un árbol. se aprenden varios árboles y a la hora de clasificar
#   nuevos ejemplos, se devuelve la clasificación mayoritaria.
# - Cada uno de esos árboles no se aprende con el conjunto de entrenamiento original, sino
#   con una muestra de ejemplos, obtenido seleccionado los ejemplos aleatoriamente del 
#   conjunto total, CON REEMPLAZO. Además, durante el aprendizaje y en cada nodo, no se usan todos
#   los atributos sino un sunconjunto de ellos obtenidos aleatoriamente (el mismo para todo el árbol). 

# NOTA IMPORTANTE: En la versión estándar del algoritmo Random Forest, el subconjunto de
# atributos a considerar se sortea EN CADA NODO de los árboles que se aprenden. Sin
# embargo, en nuestro caso, como vamos a usar la clase ArbolDecision del ejercicio
# anterior, se va usar el mismo subconjunto de atributos EN CADA ÁRBOL APRENDIDO.

# Concretando, se pide implementar una clase RandomForest con la siguiente estructura:


# class RandomForest:
#     def __init__(self, n_arboles=5,prop_muestras=1.0,
#                        min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):
#         .......                   

#     def entrena(self, X, y):
#         .......

#     def clasifica(self, X):
#         .......
    
# Los argumentos del constructor son:

# - n_arboles: el número de árboles que se van a obtener para el clasificador.
# - n_muestras: el número de ejemplos a muestrear para el aprendizaje de cada árbol.
# - El resto de argumentos son los mismos que en el ejercicio anterior, y se usan en el
#   aprendizaje de cada árbol.


# Ejemplos:
# *********

# VOTOS:
# ------

# >>> clf_votos_rf=RandomForest(n_arboles=10,min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=6,prop_umbral=0.8)
# >>> clf_votos_rf.entrena(Xe_votos, ye_votos)
# >>> rendimiento(clf_votos_rf,Xe_votos,ye_votos)
# 0.9517241379310345
# >>> rendimiento(clf_votos_rf,Xp_votos,yp_votos)
# 0.9586206896551724


# >>> clf_cancer_rf = RandomForest(n_arboles=15,min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
# >>> clf_cancer_rf.entrena(Xev_cancer, yev_cancer)
# >>> rendimiento(clf_cancer_rf,Xev_cancer,yev_cancer)
# 1.0
# >>> rendimiento(clf_cancer_rf,Xp_cancer,yp_cancer)
# 0.9911504424778761


#------------------------------------------------------------------------------




class RandomForest:
    def __init__(self, n_arboles=5,prop_muestras=1.0,
                       min_ejemplos_nodo_interior=5, max_prof=10,n_atrs=10,prop_umbral=1.0):

        self.n_arboles = n_arboles
        self.prop_muestras = prop_muestras
        self.min_ejemplos_nodo_interior = min_ejemplos_nodo_interior
        self.max_prof = max_prof
        self.n_atrs = n_atrs
        self.prop_umbral = prop_umbral
        
        self.arboles = []


    def entrena(self, X, y):
        self.arboles = []
        num_ejemplos = X.shape[0]
        n_muestras = int(np.round(num_ejemplos * self.prop_muestras))
        
        for _ in range(self.n_arboles):
            indices_aleatorios = np.random.choice(num_ejemplos, n_muestras, replace=True)
            X_muestreado = X[indices_aleatorios]
            y_muestreado = y[indices_aleatorios]
            
            arbol = ArbolDecision(
                min_ejemplos_nodo_interior=self.min_ejemplos_nodo_interior,
                max_prof=self.max_prof,
                n_atrs=self.n_atrs,
                prop_umbral=self.prop_umbral
            )
            arbol.entrena(X_muestreado, y_muestreado)
            self.arboles.append(arbol)





    def clasifica(self, X):
        votos_arboles = []
        for arbol in self.arboles:
            votos_arboles.append(arbol.clasifica(X))
            
        matriz_votos = np.array(votos_arboles)
        predicciones_finales = []
        
        for i in range(X.shape[0]):
            votos_pasajero = matriz_votos[:, i]
            clases, conteos = np.unique(votos_pasajero, return_counts=True)
            clase_ganadora = clases[np.argmax(conteos)]
            predicciones_finales.append(clase_ganadora)
            
        return np.array(predicciones_finales)


















# =========================================
# EJERCICIO 4: AJUSTANDO LOS CLASIFICADORES
# =========================================

# En este ejercicio vamos a tratar de obtener buenos clasificadores para los 
# los siguientes conjuntos de datos: IMDB, credito, AdultDataset y dígitos.

# ---------------------------
# 4.1 PREPARANDO LOS DATASETS     
# ---------------------------

# Excepto a IMDB, que ya se carga cuando se ejecuta carga_datos.py, el resto 
# tendremos que hacer antes algún preprocesado:
    
# - En X_credito, los atributos son categóricos, así que hay que transformarlos 
#   en numéricos para que se puedan usar con nuestros árboles de decisión. 
#   En el caso de árboles de decisión no es necesario hacer "one hot encoding",
#   sino que basta con codificar los valores de los atributos con números naturales
#   Para ello, SE PIDE USAR el OrdinalEncoder de sklearn.preprocessing (ver manual). 
#   Será necesario también separar en conjunto de prueba y de entrenamiento y
#   validación. 

# - El dataset AdultDataset nos viene es un archivo csv. Cargarlo con 
#   read_csv de pandas, separarlo en entrenamiento y prueba 
#   y aplicarle igualmente OrdinalEncoder, pero sólo a las características desde la 
#   quinta en adelante (ya que las cuatro primeras columnas ya son numéricas). 

# - El dataset de dígitos los podemos obtener a partir de los datos que están en 
#   la carpeta datos/digitdata que se suministra.  Cada imagen viene dada por 28x28
#   píxeles, y cada pixel vendrá representado por un caracter "espacio en
#   blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#   (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#   (es decir, no distinguiremos entre el borde y el interior). En cada
#   conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#   clasificaciones de cada imagen (es decir, el número que representan) vienen
#   en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#   funciones python que lean esos ficheros y obtengan los datos en el mismo
#   formato numpy en el que los necesita el clasificador. 
#   Los datos están ya separados en entrenamiento, validación y prueba. 

from sklearn.preprocessing import OrdinalEncoder
import pandas as pd




#   Se pide incluir aquí las definiciones y órdenes necesarias para definir
#   las siguientes variables, con los datasets anteriores como arrays de numpy.
from sklearn.model_selection import train_test_split

# * X_train_credito, y_train_credito, X_test_credito, y_test_credito
#   conteniendo el dataset de crédito con los atributos numñericos:

# convierte clases a numeros 
codificador_credito = OrdinalEncoder()
X_credito_num = codificador_credito.fit_transform(X_credito)

X_train_val_cred, X_test_credito, y_train_val_cred, y_test_credito = train_test_split(
    X_credito_num, y_credito, test_size=0.2, random_state=42
)

X_train_credito, X_valid_credito, y_train_credito, y_valid_credito = train_test_split(
    X_train_val_cred, y_train_val_cred, test_size=0.25, random_state=42
)




# * X_train_adult, y_train_adult, X_test_adult, y_test_adult
#   conteniendo el AdultDataset con los atributos numéricos:


df_adult = pd.read_csv("datos/adultDataset.csv")

X_caract = df_adult.iloc[:, :-1].values
y_numeros = df_adult.iloc[:, -1].values

codificador_adult = OrdinalEncoder()
X_caract[:, 4:] = codificador_adult.fit_transform(X_caract[:, 4:])

X_train_val_ad, X_test_adult, y_train_val_ad, y_test_adult = train_test_split(
    X_caract, y_numeros, test_size=0.2, random_state=42
)

X_train_adult, X_valid_adult, y_train_adult, y_valid_adult = train_test_split(
    X_train_val_ad, y_train_val_ad, test_size=0.25, random_state=42
)





# * X_train_dg, y_train_dg, X_valid_dg, y_valid_dg, X_test_dg, y_test_dg
#   conteniendo el dataset de los dígitos escritos a mano:
    

def leer_fichero_imagenes(ruta):
    imagenes = []
    imagen_actual = []
    lineas_contadas = 0
    
    with open(ruta, 'r') as f:
        for linea in f:
            linea = linea.replace('\n', '').replace('\r', '')
        
            for caracter in linea:
                if caracter == ' ':
                    imagen_actual.append(0)
                else:
                    imagen_actual.append(1) 
            
            lineas_contadas += 1
            if lineas_contadas == 28:
                imagenes.append(imagen_actual)
                imagen_actual = []
                lineas_contadas = 0
    return np.array(imagenes)

def leer_fichero_numeros(ruta):
    numeros = []
    
    with open(ruta, 'r') as f:
        for linea in f:
            linea = linea.strip()
    
            if linea:
                numero = int(linea)
                numeros.append(numero)
                
    return np.array(numeros)


X_train_dg = leer_fichero_imagenes("datos/digitdata/trainingimages")
y_train_dg = leer_fichero_numeros("datos/digitdata/traininglabels")

X_valid_dg = leer_fichero_imagenes("datos/digitdata/validationimages")
y_valid_dg = leer_fichero_numeros("datos/digitdata/validationlabels")

X_test_dg  = leer_fichero_imagenes("datos/digitdata/testimages")
y_test_dg  = leer_fichero_numeros("datos/digitdata/testlabels")






# -----------------------------
# 4.2 AJUSTE DE HIPERPARÁMETROS     
# -----------------------------

# En nuestra implementación de RandomForest tenemos los siguientes 
# hiperparámetros: 

# n_arboles
# prop_muestras
# min_ejemplos_nodo_interior
# max_prof
# n_atrs
# prop_umbral

# Se trata ahora de encontrar, en cada dataset, una buena combinación de valores para esos 
# hiperparámetros, tratando de obtener un buen rendimiento de los clasificadores. Hacerlo
# usando un conjunto de validación: según se ha visto en la teoría, esto consiste en particionar  
# en entrenamiento, validación y prueba, entrenando por cada combinación de hiperparámetros 
# con el conjunto de entrenamiento y evaluando el rendimiento en validación. El entrenamiento final 
# con la mejor combinación ha de hacerse en la unión de entrenamiento y validación.
    

# NO ES NECESARIO ser demasiado exhaustivo, basta con probar algunas combinaciones, 
    # pero sí es importante describir el proceso realizado y las mejores combinaciones 
    # encontradas en cada caso. 
# DEJAR ESTE APARTADO COMENTADO, para que no se ejecuten las pruebas realizadas cuando se cargue
# el archivo. 

# ----------------------------




# ------------------------------------------------------------------------------
# DATASET: CREDITO 
# ------------------------------------------------------------------------------


"""
print("PRUEBAS PARA CRÉDITO")

# 1. PROBAMOS LA COMBINACIÓN A (Bosque pequeño con árboles cortos)
rf_A = RandomForest(n_arboles=3, max_prof=4, n_atrs=5, min_ejemplos_nodo_interior=5)

# Lo entrenamos con los datos de estudiar (Train)
rf_A.entrena(X_train_credito, y_train_credito)

# Probamos con el conjunto de validación
predicciones_A = rf_A.clasifica(X_valid_credito)
acierto_A = np.mean(predicciones_A == y_valid_credito)
print("Combinación A (3 árboles, prof 4) -> Nota en Validación:", acierto_A)


# 2. PROBAMOS LA COMBINACIÓN B (Bosque más grande con más profundidad)
rf_B = RandomForest(n_arboles=10, max_prof=8, n_atrs=5, min_ejemplos_nodo_interior=5)

# Lo entrenamos con los mismos datos de estudiar (Train)
rf_B.entrena(X_train_credito, y_train_credito)

# Probamos con el conjunto de validación
predicciones_B = rf_B.clasifica(X_valid_credito)
acierto_B = np.mean(predicciones_B == y_valid_credito)

print("Combinación B (10 árboles, prof 8) -> Nota en Validación:", acierto_B)


# 3. Juntamos las matrices (las filas de train y las filas de validación)
X_completo_credito = np.vstack((X_train_credito, X_valid_credito))
y_completo_credito = np.concatenate((y_train_credito, y_valid_credito))

# Creamos el modelo con la combinación de B
rf_definitivo = RandomForest(n_arboles=10, max_prof=8, n_atrs=5, min_ejemplos_nodo_interior=5)
rf_definitivo.entrena(X_completo_credito, y_completo_credito)

predicciones_test = rf_definitivo.clasifica(X_test_credito)
acierto_test_final = np.mean(predicciones_test == y_test_credito)
print("Nota en el conjunto de TEST: ", acierto_test_final)

"""

# ==============================================================================
# 1. MEJOR COMBINACIÓN: La Combinación B (10 árboles, prof 8) resultó ganadora 
#    con un 91.53% de acierto frente al 88.46% de la Combinación A. Esto demuestra 
#    que un bosque más denso y profundo reduce con mayor eficacia el sesgo del 
#    modelo sin incurrir en sobreajuste gracias al promediado del Bagging.
#
# 2. RENDIMIENTO FINAL: El modelo definitivo logró un 87.69% de precisión en TEST. 
#    La consistencia entre la nota de validación y la de test certifica que el 
#    clasificador tiene una excelente capacidad de generalización en entornos reales.
# ==============================================================================







# ------------------------------------------------------------------------------
# DATASET: ADULT 
# ------------------------------------------------------------------------------

"""
print("PRUEBAS PARA ADULT")
# 1. Probamos combinación A: Bosque pequeño y rápido
rf_adult_A = RandomForest(n_arboles=4, max_prof=4, n_atrs=6, min_ejemplos_nodo_interior=5)
rf_adult_A.entrena(X_train_adult, y_train_adult)
acierto_adult_A = np.mean(rf_adult_A.clasifica(X_valid_adult) == y_valid_adult)
print("Combinación A (4 árboles, prof 4) -> Validación:", acierto_adult_A )

# 2. Probamos combinación B: Bosque más grande
rf_adult_B = RandomForest(n_arboles=12, max_prof=8, n_atrs=8, min_ejemplos_nodo_interior=5)
rf_adult_B.entrena(X_train_adult, y_train_adult)
acierto_adult_B = np.mean(rf_adult_B.clasifica(X_valid_adult) == y_valid_adult)
print("Combinación B (12 árboles, prof 8) -> Validación:", acierto_adult_B)

# 3. Juntamos las matrices (las filas de train y las filas de validación) 
X_completo_adult = np.vstack((X_train_adult, X_valid_adult))
y_completo_adult = np.concatenate((y_train_adult, y_valid_adult))

rf_def_adult = RandomForest(n_arboles=12, max_prof=8, n_atrs=8, min_ejemplos_nodo_interior=5)
rf_def_adult.entrena(X_completo_adult, y_completo_adult)
acierto_adult_test = np.mean(rf_def_adult.clasifica(X_test_adult) == y_test_adult)
print("Nota en el conjunto de TEST: ", acierto_adult_test)

"""

# ==============================================================================
# 1. MEJOR COMBINACIÓN: La Combinación B (12 árboles, prof 8) resultó ganadora 
#    con un 85.17% de acierto frente al 80.68% de la Combinación A. Esto demuestra 
#    que la complejidad socioeconómica del censo requiere árboles con más pisos 
#    de profundidad para reducir el subajuste (underfitting) y extraer mejores patrones.
#
# 2. RENDIMIENTO FINAL: El modelo definitivo logró un 85.26% de precisión en TEST. 
#    Que la nota se mantenga tan sólida e incluso suba una décima respecto a validación 
#    certifica que el bosque es tremendamente estable y aprovecha muy bien los datos extra.
# ==============================================================================






# ------------------------------------------------------------------------------
# 2. DATASET: DÍGITOS 
# ------------------------------------------------------------------------------

"""
print("PRUEBAS PARA DIGITOS")
# 1. Probamos combinación A: Bosque pequeño y rápido
rf_dg_A = RandomForest(n_arboles=5, max_prof=5, n_atrs=15, min_ejemplos_nodo_interior=4)
rf_dg_A.entrena(X_train_dg, y_train_dg)

predicciones_dg_A = rf_dg_A.clasifica(X_valid_dg)               
acierto_dg_A = np.mean(predicciones_dg_A == y_valid_dg)
print("Combinación A (5 árboles, prof 5) -> Validación:", acierto_dg_A)


# 2. Probamos combinación B: Bosque más grande
rf_dg_B = RandomForest(n_arboles=15, max_prof=10, n_atrs=30, min_ejemplos_nodo_interior=4)
rf_dg_B.entrena(X_train_dg, y_train_dg)

predicciones_dg_B = rf_dg_B.clasifica(X_valid_dg)               
acierto_dg_B = np.mean(predicciones_dg_B == y_valid_dg)
print("Combinación B (15 árboles, prof 10) -> Validación:", acierto_dg_B)


# 3. Juntamos las matrices (las filas de train y las filas de validación) 
print("-> Entrenando modelo definitivo de Dígitos con la combinación ganadora...")
X_completo_dg = np.vstack((X_train_dg, X_valid_dg))
y_completo_dg = np.concatenate((y_train_dg, y_valid_dg))

rf_def_dg = RandomForest(n_arboles=15, max_prof=10, n_atrs=30, min_ejemplos_nodo_interior=4)
rf_def_dg.entrena(X_completo_dg, y_completo_dg)

predicciones_dg_test = rf_def_dg.clasifica(X_test_dg)            
acierto_dg_test = np.mean(predicciones_dg_test == y_test_dg)
print("Nota en el conjunto de TEST: ", acierto_dg_test)


# ==============================================================================
# 1. MEJOR COMBINACIÓN: La Combinación B (15 árboles, prof 10) resultó ganadora 
#    con un 82.60% de acierto frente al 45.60% de la Combinación A. Al clasificar 
#    imágenes por píxeles, un mayor muestreo de atributos asegura capturar los 
#    trazos clave de cada número sin confundir dígitos similares.
#
# 2. RENDIMIENTO FINAL: El modelo definitivo logró un 78.80% de precisión en TEST. 
#    La consistencia en una matriz de datos tan grande demuestra que la aleatoriedad 
#    del RandomForest dota al clasificador de una gran robustez ante ruidos en el trazo.
# ==============================================================================

"""










# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************

# EJEMPLOS DE PRUEBA

# LAS SIGUIENTES LLAMADAS SERÁN EJECUTADAS POR EL PROFESOR EL DÍA DE LA PRESENTACIÓN.
# UNA VEZ IMPLEMENTADAS LAS DEFINICIONES Y FUNCIONES NECESARIAS
# Y REALIZADOS LOS AJUSTES DE HIPERPARÁMETROS, 
# DEJAR COMENTADA CUALQUIER LLAMADA A LAS FUNCIONES QUE SE TENGA EN ESTE ARCHIVO 
# Y DESCOMENTAR LAS QUE VIENEN A CONTINUACIÓN.
# UNA VEZ HECHO ESO, ENTREGAR TAMBIÉN UN ARCHIVO trabajo_aia_25_26_parte_I_sc.py
# SIN COMENTARIOS AL CÓDIGO. 


# EN EL APARTADO FINAL DE "RENDIMIENTOS FINALES RANDOM FOREST", USAR LA MEJOR COMBINACIÓN DE 
# HIPERPARÁMETROS QUE SE HAYA OBTENIDO EN CADA CASO, EN LA FASE DE AJUSTE DEL EJERCICIO 4

# El ARCHIVO trabajo_aia_23_24_parte_I_sc.py SERÁ CARGADO POR EL PROFESOR, 
# TENIENDO EN LA MISMA CARPETA LOS ARCHIVOS OBTENIDOS
# DESCOMPRIMIENDO datos_trabajo_aia.zip.
# ES IMPORTANTE QUE LO QUE SE ENTREGA SE PUEDA CARGAR SIN ERRORES Y QUE SE EJECUTEN LOS 
# EJEMPLOS QUE VIENEN A CONTINUACIÓN. SI ALGUNO DE LOS EJERCICIOS NO SE HA REALIZADO 
# O DEVUELVE ALGÚN ERROR, DEJAR COMENTADOS LOS CORRESPONDIENTES EJEMPLOS, 
# PARA EViTAR LOS ERRORES EN LA CARGA Y EJECUCIÓN.   



# *********** DESCOMENTAR A PARTIR DE AQUÍ

print("************ PRUEBAS EJERCICIO 1:")
print("**********************************\n")
Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
print("Partición votos: ",y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
print("Proporción original en votos: ",np.unique(y_votos,return_counts=True))
print("Estratificación entrenamiento en votos: ",np.unique(ye_votos,return_counts=True))
print("Estratificación prueba en votos: ",np.unique(yp_votos,return_counts=True))
print("\n")

Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
print("Proporción original en cáncer: ", np.unique(y_cancer,return_counts=True))
print("Estratificación entr-val en cáncer: ",np.unique(yev_cancer,return_counts=True))
print("Estratificación prueba en cáncer: ",np.unique(yp_cancer,return_counts=True))
Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer,return_counts=True))
print("Estratificación validación cáncer: ",np.unique(yv_cancer,return_counts=True))
print("\n")

Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
print("Estratificación entrenamiento crédito: ",np.unique(ye_credito,return_counts=True))
print("Estratificación prueba crédito: ",np.unique(yp_credito,return_counts=True))
print("\n\n\n")





print("************ PRUEBAS EJERCICIO 2:")
print("**********************************\n")

clf_titanic = ArbolDecision(max_prof=3,min_ejemplos_nodo_interior=5,n_atrs=3)
clf_titanic.entrena(X_train_titanic, y_train_titanic)
clf_titanic.imprime_arbol(["Pclass", "Mujer", "Edad"],"Partido")
rend_train_titanic = rendimiento(clf_titanic,X_train_titanic,y_train_titanic)
rend_test_titanic = rendimiento(clf_titanic,X_test_titanic,y_test_titanic)
print(f"****** Rendimiento DT titanic train: {rend_train_titanic}")
print(f"****** Rendimiento DT titanic test: {rend_test_titanic}\n\n\n\n ")




clf_votos = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=5,n_atrs=16)
clf_votos.entrena(Xe_votos, ye_votos)
nombre_atrs_votos=[f"Votación {i}" for i in range(1,17)]
clf_votos.imprime_arbol(nombre_atrs_votos,"Partido")
rend_train_votos = rendimiento(clf_votos,Xe_votos,ye_votos)
rend_test_votos = rendimiento(clf_votos,Xp_votos,yp_votos)
print(f"****** Rendimiento DT votos en train: {rend_train_votos}")
print(f"****** Rendimiento DT votos en test:  {rend_test_votos}\n\n\n\n")


X_train_iris, X_test_iris, y_train_iris, y_test_iris = particion_entr_prueba(X_iris, y_iris, test=0.20)
clf_iris = ArbolDecision(max_prof=3,n_atrs=4)
clf_iris.entrena(X_train_iris, y_train_iris)
clf_iris.imprime_arbol(["Long. Sépalo", "Anch. Sépalo", "Long. Pétalo", "Anch. Pétalo"],"Clase")
rend_train_iris = rendimiento(clf_iris,X_train_iris,y_train_iris)
rend_test_iris = rendimiento(clf_iris,X_test_iris,y_test_iris)
print(f"********************* Rendimiento DT iris train: {rend_train_iris}")
print(f"********************* Rendimiento DT iris test: {rend_test_iris}\n\n\n\n ")





clf_cancer = ArbolDecision(min_ejemplos_nodo_interior=3,max_prof=10,n_atrs=15)
clf_cancer.entrena(Xev_cancer, yev_cancer)
nombre_atrs_cancer=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error',
        'fractal dimension error', 'worst radius', 'worst texture',
        'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points',
        'worst symmetry', 'worst fractal dimension']
clf_cancer.imprime_arbol(nombre_atrs_cancer,"Es benigno")
rend_train_cancer = rendimiento(clf_cancer,Xev_cancer,yev_cancer)
rend_test_cancer = rendimiento(clf_cancer,Xp_cancer,yp_cancer)
print(f"***** Rendimiento DT cancer en train: {rend_train_cancer}")
print(f"***** Rendimiento DT cancer en test: {rend_test_cancer}\n\n\n")



print("************ RENDIMIENTOS FINALES RANDOM FOREST")
print("************************************************\n")


# # ATENCIÓN: EN CADA CASO, INCORPORAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS 
# # QUE SE HA OBTENIDO EN EL PROCESO DE AJUSTE



print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE IMDB:")
RF_IMDB = RandomForest(n_arboles=45, prop_muestras=0.6, max_prof=25, min_ejemplos_nodo_interior=3, n_atrs=20)
RF_IMDB.entrena(X_train_imdb,y_train_imdb) 
print("Rendimiento RF entrenamiento sobre imdb: ",rendimiento(RF_IMDB,X_train_imdb,y_train_imdb))
print("Rendimiento RF test sobre imdb: ",rendimiento(RF_IMDB,X_test_imdb,y_test_imdb))
print("\n")




print("==== MEJOR RENDIMIENTO RANDOM FOREST SOBRE CRÉDITO:")

RF_CREDITO = RandomForest(n_arboles=50, prop_muestras=0.8, max_prof=8, min_ejemplos_nodo_interior=5, n_atrs=4)# ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
RF_CREDITO.entrena(X_train_credito,y_train_credito) 
print("Rendimiento RF entrenamiento sobre crédito: ",rendimiento(RF_CREDITO,X_train_credito,y_train_credito))
print("Rendimiento RF  test sobre crédito: ",rendimiento(RF_CREDITO,X_test_credito,y_test_credito))
print("\n")


print("==== MEJOR RENDIMIENTO RF SOBRE ADULT:")

RF_ADULT = RandomForest(n_arboles=40, prop_muestras=0.7, max_prof=12, min_ejemplos_nodo_interior=10, n_atrs=6) # ATENCIÓN: incorporar aquí los mejores valores de los parámetros tras el ajuste
RF_ADULT.entrena(X_train_adult,y_train_adult) 
print("Rendimiento RF  entrenamiento sobre adult: ",rendimiento(RF_ADULT,X_train_adult,y_train_adult))
print("Rendimiento RF  test sobre adult: ",rendimiento(RF_ADULT,X_test_adult,y_test_adult))
print("\n")


print("==== MEJOR RENDIMIENTO RL SOBRE DIGITOS:")
RF_DG = RandomForest(n_arboles=50, prop_muestras=0.9, max_prof=28, min_ejemplos_nodo_interior=2, n_atrs=30) # ATENCIÓN: incorporar aquí los mejores valors de losparámetros tras el ajuste
RF_DG.entrena(X_train_dg,y_train_dg)
print("Rendimiento RF entrenamiento sobre dígitos: ",rendimiento(RF_DG,X_train_dg,y_train_dg))
print("Rendimiento RF validación sobre dígitos: ",rendimiento(RF_DG,X_valid_dg,y_valid_dg))
print("Rendimiento RF test sobre dígitos: ",rendimiento(RF_DG,X_test_dg,y_test_dg))