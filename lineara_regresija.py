import pandas as pd # datu apstrāde
import numpy as np  # darbs ar masīviem
from termcolor import colored as cl # teksta izvade
import matplotlib.pyplot as plt # vizualizācija
import seaborn as sb # vizualizācija
import pickle

from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm
from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric


def sagatavot_datus(datne, kolonnas_x, kolonnas_y):

    df = pd.read_csv(datne)

    # reizēm nepieciešams izmest datus, kur ir tukšas vērtības
    # df.dropna(inplace = True)
    # vai arī aizpildīt tukšos datus ar 0
    # df = df.fillna(0)

    # # reizem nepieciešams pārvērst datu tipus
    # df = parverst_kolonnu(df, kolonna)

    # neatkarīgās (ieejas/avota) vērtības, sauc par X_var
    X_var = df[kolonnas_x].values 
    # atkarīgās (izejas) vērtības, sauc par y_var
    y_var = df[kolonnas_y].values

    # sadalām datus trenēšanā izmantojamos un pārbaudei atstātos
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)

    return(X_train, X_test, y_train, y_test)


def parverst_kolonnu(df, kolonna):
    # pārvēršam visu par Integer, jo tā vajag LinearRegression
    df[kolonna] = pd.to_numeric(df[kolonna], errors = 'coerce')
    df[kolonna] = df[kolonna].astype('int64')
    return df


def modela_kvalitate(y_test, resultats):
    print(cl('Explained Variance Score: {}'.format(evs(y_test, resultats)), attrs = ['bold']))
    print(cl('R-Squared: {}'.format(r2(y_test, resultats)), attrs = ['bold']))


def saglabat_modeli(datne, modelis):
    with open(datne, 'wb') as f:
        s = pickle.dumps(modelis)
        f.write(s)


def ieladet_modeli(datne):
    with open(datne, 'rb') as f:
        s = f.read()
        modelis = pickle.loads(s)
    return modelis


def trenet_modeli(modelis, X_train, y_train, X_test, saglabat=None):
    modelis = LinearRegression()
    modelis.fit(X_train, y_train)
    if saglabat:
        saglabat_modeli(saglabat, modelis)
    resultats = modelis.predict(X_test)
    return resultats


def prognozejam_rezultatu(modelis, dati):
    rezultats = modelis.predict(dati)
    return rezultats


datne1 = 'dati/auto_simple.csv'
kol_x1 = ['Volume','Weight']
kol_y1 = 'CO2'

datne2 = 'dati/auto_imports.csv'
kol_x2 = ['wheel-base','length','engine-size','city-mpg']
kol_y2 = 'price'

# Sagatavojam datus no datnes
X_train, X_test, y_train, y_test = sagatavot_datus(datne2, kol_x2, kol_y2)


# vienkārša lineārā regresija
modelis = LinearRegression()
# Citi algoritmi ko var lietot:
# # 2. Ridge
# ridge = Ridge(alpha = 0.5)
# # 3. Lasso
# lasso = Lasso(alpha = 0.01)
# # 4. Bayesian
# bayesian = BayesianRidge()
# # 5. ElasticNet
# en = ElasticNet(alpha = 0.01)
rezultats = trenet_modeli(modelis, X_train, y_train, X_test)
# # Ja gribam saglabāt modeli datnē
# rezultats = trenet_modeli(modelis, X_train, y_train, X_test, "modelis.pickle")
modela_kvalitate(y_test, rezultats)

print("Ielādējam modeli no datnes")
modelis2 = ieladet_modeli("modelis.pickle")
modelis2.fit(X_train, y_train)
rezultats2 = modelis2.predict(X_test)
modela_kvalitate(y_test, rezultats2)


# Lietojam modeli, lai prognozetu rezultātu
dati1 = [1500,1140]
dati1_rez = 105
dati2 = [99.80,176.60,109,24]
dati2_rez = 13950

prognoze = prognozejam_rezultatu(modelis2, [dati2])
print(prognoze, dati2_rez)
