# -*- coding: utf-8 -*-
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Importing the dataset- AIRBNB rentals in Amsterdam 
dataset = pd.read_csv('airbnbamsterdam1.csv')
X = dataset.iloc[:,:].values

dataset.shape

dataset.head(10)

dataset.dtypes

#running descriptive statistics on numerical variable price

dataset['price'].describe()

#Facetgrid of room type and prices ( of rooms priced till 500)
g = sns.FacetGrid(dataset, col="room_type",xlim=500) 
g.map(sns.distplot, "price")
sns.plt.show()

#pairplots useful to plot different plots together on various variables 
g = sns.pairplot(dataset[["accommodates", "price" ,"neighborhood"]], hue="neighborhood", kind="hist")

for ax in g.axes.flat: 
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    

