#!/usr/bin/env python
# coding: utf-8

# In[1]:



#Por  Martinez Sanchez Jose Manuel.
                          
                          


# In[2]:


import re
import numpy as np 
import pandas as pd 

print(__doc__)
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
from sklearn.tree import DecisionTreeRegressor

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# ## Cargando los conjunto de datos proporcionados.

# In[3]:


test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")


# ## Analisis exploratorio de datos.

# In[4]:


train_df.info()
total = train_df.isnull().sum().sort_values(ascending=False)
p1 = train_df.isnull().sum()/train_df.isnull().count()*100
p2 = (round(p1, 1)).sort_values(ascending=False)
datos_Faltantes = pd.concat([total, p2], axis=1, keys=['Observaciones', '% Perdidas'])
datos_Faltantes.head(5)


# Observamos 12 variables, de las cuales las variables "Cabin" y "Age" tienen el mayor  porcentaje de valores vacios. 
# Este hecho nos permite determinar desde un inicio cuales valiables deben ser analizadas con especial atencion en el pretratamiento de datos.
# 
# En este contexto es pertinente explorar la influencia de cada variable en la tasa de supervivencia.

# In[5]:


survived = 'Sobrevive'
not_survived = 'No sobrevive'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Mujeres')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Hombres')


# In[6]:


FacetGrid = sns.FacetGrid(train_df, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


# 

# In[7]:


sns.barplot(x='Pclass', y='Survived', data=train_df)


# Observamos un mayor rango de supervivencia para hombres y mujeres cuando se encuentran en un rango de edad de entre 18 y 40 años, sin embargo parece existir una relacion mas notable entre el puerto de embarcacion y el genero, por ejemplo, observamos que los hombres que embarcaron en el puerto C sobreviven mucho mas que si se encontraran en el puerto Q o S. De igual manera las mujeres en el puerto Q y en el puerto S tienen una mayor tasa de supervivencia.
# 
# Ademas de lo anterior, es especialmente evidente que la tasa de supervivencia es mas elevada para un pasjero de priemera clase. 

# In[8]:


data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df['not_alone'].value_counts()

axes = sns.factorplot('relatives','Survived', data=train_df, aspect = 2.5, )


# Observamos que la cantidad de parientes a bordo esta relacionada con la tasa de sobrevivencia solo cuando se tienen de 1-3 o exactamente 6 parientes.

# ## Preprocesamiento !
# De acuerdo a las observaciones anteriores podemos determinar dos objetivos de procesameinto de datos. Eliminar las variables irrelevantes para el analisis y normalizar los conjuntos de datos de entrada.
# 
# Para el primer objetivo eliminamos la varaible PassengerId exclusivamente del conjunto de datos de entrenamiento. Para el segundo objetivo convertimos la variable 'Cabin' a una codificacion numerica y asignamos valores de edad aleatorios  para los valores de edad vacios.

# In[9]:


train_df = train_df.drop(['PassengerId'], axis=1)
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)


train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


data = [train_df, test_df]

"""
Para la edad debemos calcular los valores aleatorios de edad en base a los valores de la media y Stdev para evitar sesgos.
"""


for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()

#La variable Emarked presenta solo dos valores vacios por lo que asignamos el mas comun.

train_df['Embarked'].describe()
common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

print(data) 


# ### Transformacion de variables categoricas..
# Observamos la existencia de 4 variables categoricas (Name, Sex, Ticket y Embarked), por lo que procedemos a homgenizar tipos y transformar categorias en variabels numericas.
# 

# In[10]:


#Fare  Conversion de Float -> Int
data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    
#Name.

"""
En este caso es escencial crear una nueva caracteristica a partir de los datos no numericos.
"""
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)
    # NaN
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)



#Sex
genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
    

#Ticket
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

#Emarked
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
    

#Creando la categoria Edad.
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# 
train_df['Age'].value_counts()


#Modificando la catergoria Fare
data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
    
#Asociamos el valor de la edad con la clase del pasajero    
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

#Asociamos la tarifa por pasajero (Para observar la reacion entre sus acompanantes, la tarifa y tasa de supervivencia)    
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)


# ## Evaluacion de distintos modelos de ML
# 
# El ejercicio especifica la comparacion en almenos 2 modelos de ML, la propuesta presentada es la siguiente:
# 
# 
# Regresion logistica.
# 
# Decenso estocastico.
# 
# Random Forest.

# In[11]:


#Creando Datasets de entrenamiento.
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

#Regresion logistica:
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

#Decenso estocastico
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)


#Random forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# ## Evaluando resultados.

# In[12]:


results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Stochastic Gradient Decent',
              'Random Forest'],
    'Score': [acc_log, acc_sgd,
              acc_random_forest]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)



# In[13]:


results.plot.bar(x='Model', y='Score', rot=0)


# ### Observamos que el modelo con mejor rendimiento (score = Predictive Accuracy)

# In[14]:


plt.plot(Y_train, color='blue', linewidth=3)
plt.plot(Y_prediction, color='yellow', linewidth=3)
#plt.xticks(())
#plt.yticks(())
plt.show()


# ## Validacion.

# In[15]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Media:", scores.mean())
print("Desviacion estandar:", scores.std())

influence = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
influence = influence.sort_values('importance',ascending=False).set_index('feature')
influence.head(15)

influence.plot.bar()


# In[16]:


from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)


# In[29]:


from sklearn.metrics import classification_report
y_true = Y_train
y_pred = predictions
target_names = ['class 0', 'class 1']
print(classification_report(y_true, y_pred, target_names=target_names))


# ## Eliminando variables no significativas.
# 
# Observamos que las variables "not_alone" y "Parch" no son reelevantes para el sistema por lo que son eliminadas sin perdida de generalidad.

# In[18]:


train_df  = train_df.drop("not_alone", axis=1)
test_df  = test_df.drop("not_alone", axis=1)

train_df  = train_df.drop("Parch", axis=1)
test_df  = test_df.drop("Parch", axis=1)


# In[31]:


# Reentrenando Random Forest

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
percent_correct = (round(acc_random_forest,2,), "%")
print(round(acc_random_forest,2,), "%")


# In[20]:


#param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
#from sklearn.model_selection import GridSearchCV, cross_val_score
#rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
#clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
#clf.fit(X_train, Y_train)
#clf.bestparams


# In[21]:


# Testing Random Forest new hyperparams
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print("Out of the Bag score:", round(random_forest.oob_score_, 4)*100, "%")


# In[22]:



predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)


# In[30]:


#Funciones para graficar la matriz de confusion y el random forest.

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Matriz de Confusion Normalizada.'
        else:
            title = 'Matriz de Confusion son Normalizacion.'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusion Normalizada.")
    else:
        print('Matriz de Confusion son Normalizacion.')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def data_to_plotly(x):
    k = []
    
    for i in range(0, len(x)):
        k.append(x[i][0])
        
    return k


# In[24]:


plot_confusion_matrix(Y_train, predictions, classes=X_train.columns,
                      title='Confusion matrix, without normalization')


# In[25]:


# Plot normalized confusion matrix
plot_confusion_matrix(Y_train, predictions, classes=X_train.columns, normalize=True,
                      title='Normalized confusion matrix')


# In[26]:


from sklearn.metrics import classification_report
y_true = Y_train
y_pred = predictions
target_names = ['Valores predichos', 'Valores reales']
print(classification_report(y_true, y_pred, target_names=target_names))


# In[33]:



ones = np.count_nonzero(predictions)
zeros = predictions.size - ones
labels = 'Sobrevivientes', 'No Sobrevivientes'
sizes = [ones,zeros]
explode = (0, 0.1, 0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels)
ax1.axis('equal')  

plt.show()

print(percent_correct, " de clasificacion correcta" )


# In[40]:


test_df.info()
print(Y_prediction.size)


# In[54]:


print(type(Y_prediction))


# In[60]:


data = Y_prediction
pd.DataFrame(Y_prediction,data)

list_of_tuples = list(zip(test_df['PassengerId'], Y_prediction))   
final_df = pd.DataFrame(list_of_tuples, columns = ['PassengerId', 'Y_prediction'])  
final_df     


# In[59]:


final_df.to_csv("finalDF.csv", sep='\t')


# In[ ]:




