import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np


data_df = pd.read_csv('dataset_name.csv') # your dataset location
num_features = 0 # Write total features here or can be taken directly
target_var = 0 # Target variable
train_data, test_data = train_test_split(data_df, test_size=0.3)
train_x, test_x = train_data.iloc[:, :num_features], test_data.iloc[:, :num_features]
train_y, test_y = train_data.iloc[:, target_var], test_data.iloc[:, target_var]


def fitness_function(positions):
    features = np.where(positions>=0.4999)[0]
    
    if len(features) == 0:
        print("No features selected")
        return np.inf
    
    train_xf = train_x.iloc[:, features]
    
    test_xf = test_x.iloc[:, features]
    
    knn_classifier = KNeighborsClassifier(n_neighbors=7)
    
    knn_classifier.fit(train_xf, train_y)
    
    accuracy = knn_classifier.score(test_xf, test_y)
    
    alpha=0.9
    return (alpha*(1-accuracy)+(1-alpha)*((len(features)/num_features)))