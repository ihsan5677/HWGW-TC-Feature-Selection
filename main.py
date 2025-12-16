import hwgw_tc as hwgw
import obj_function as ff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from warnings import simplefilter
# ignore all future warnings.
simplefilter(action='ignore', category=FutureWarning)


fit = hwgw.HWGWTC(ff.fitness_function, 0, 1, ff.num_features, 10, 100)
selected_features = np.where(fit>0.5)[0]
print("selected features",selected_features)
fitness = ff.fitness_function(fit)

train_x = ff.train_x.iloc[:, selected_features]
test_x = ff.test_x.iloc[:, selected_features]

knn_classifier = ff.KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train_x, ff.train_y)

predicted = knn_classifier.predict(test_x)
accuracy = accuracy_score(ff.test_y, predicted)
    
print("HWGW-TC's Accuracy: ", accuracy)
print("HWGW-TC's Fitness", fitness)