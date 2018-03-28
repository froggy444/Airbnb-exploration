# -*- coding: utf-8 -*-

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('airbnbamsterdam1.csv')
X = dataset.iloc[:, 6:9].values
y = dataset.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

np.any(np.isnan(X))
np.any(np.isfinite(X))



# Taking care of missing data
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:3])
X[:, 0:3] = imputer.transform(X[:, 0:3])

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result- satis,acc, nrooms
y_pred = regressor.predict(X)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Overall satisfaction')
plt.ylabel('Price')
plt.show()

sns.pairplot(dataset, x_vars=['overall_satisfaction', 'bedrooms'], y_vars=[y_pred],
             size=2, aspect=0.1, kind="reg");
             
sns.regplot(x="overall_satisfaction", y=y_pred, data=dataset,fit_reg=True,y_jitter=0.03,scatter=False)

             
sns.regplot(x="bedrooms", y=y_pred, data=dataset)


sns.regplot(x="accommodates", y=y_pred, data=dataset)

# Kernel SVM



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('airbnbamsterdam1.csv')

X = dataset.iloc[:,6:10].values
y = dataset.iloc[:, 4].values
y_unique=np.unique(y)






# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:5])
X[:, 0:5] = imputer.transform(X[:, 0:5])


#from sklearn.preprocessing import Imputer
#yimputer=Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#yimputer = yimputer.fit(y)
#y = yimputer.transform(y)


#Encoding the categorical variable neighborhood
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)





# Splitting the dataset into the Training set and Test set

# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



import seaborn as sns
df_cm = pd.DataFrame(cm, range(19),
                  range(19))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},xticklabels=y_unique,yticklabels=y_unique)# font size

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#categories 

#datasetnew['n_type']=datasetnew['neighborhood']
#datasetnew.drop(['neighborhood'],axis=1)

dataset['neighborhood']=dataset['n_type']
dataset.head()

HighlyPopular = ["Centrum West","Centrum Oost","De Baarsjes / Oud West","De Pijp / Rivierenbuurt"]
Popular=["Noord-West / Noord-Midden","Westerpark","Oostelijk Havengebied / Indische Buurt","Oud Oost"]
ModeratelyPopular= ["Bos en Lommer", "Watergraafsmeer","Slotervaart","Oud Noord"]

def filter_n(n_type):
    if n_type in HighlyPopular:
        return 'Highly Popular'
    elif n_type in Popular:
        return 'Popular'
    elif n_type in ModeratelyPopular:
        return 'Moderately Popular'
    else:
        return 'Low Popularity'
    
dataset['n_type'].apply(filter_n)
