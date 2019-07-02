##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 11th 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

#This file provides a classical benchmark for the same problem our quantum classifier is tackling
#We use a standard classifier, a SVM by scikit learn

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from data_gen import data_generator
import numpy as np

problem = 'squares'
print(problem)
data, drawing = data_generator(problem) #name for the problem

train_data = data[:200]
test_data = data[200:]

X_train = []
Y_train = []
for d in train_data:
    x, y = d
    X_train.append(x)
    Y_train.append(y)
    
X_test = []
Y_test = []
for d in test_data:
    x, y = d
    X_test.append(x)
    Y_test.append(y)
    

text_file_nn = open('classical_benchmark/' + problem + '_nn', mode='w')
text_file_svc = open('classical_benchmark/' + problem + '_svc', mode='w')
nn = 0
svc = 0
for i in range(10):
    clf = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',
                    solver = 'lbfgs')
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    value_nn = 1 - np.sum(np.abs(pred - Y_test)) / len(Y_test)
    text_file_nn.write(str(value_nn))
    text_file_nn.write('\n')
    if value_nn > nn:
        nn = value_nn

    clf = SVC(gamma = 'auto')
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    value_svc = 1 - np.sum(np.abs(pred - Y_test)) / len(Y_test)
    text_file_svc.write(str(value_svc))
    text_file_svc.write('\n')
    if value_svc > svc:
        svc = value_svc

print('NN: ', nn)
print('SVC: ', svc)
text_file_nn.close() 
text_file_svc.close() 
