# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:56:20 2019

@author: cameron.albin
"""

import sys, os
import scipy, numpy, matplotlib, pandas, sklearn
import random
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


os.chdir('c:\\Users\\cameron.albin\\Documents\\Machine Learning Project')
dataset = pandas.read_csv('Transformer Abridged.csv')

array = dataset.values
X = array[:,0:4]
Y = array[:,6]
#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#scoring = 'accuracy'
#
#
#models = []
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('RFC', RandomForestClassifier()))
#models.append(('NB', GaussianNB()))
#
#
#print('Sales Amount:')
#results = []
#names = []
#for name, model in models:
#	kfold = model_selection.KFold(n_splits=10, random_state=seed)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)
#print('')


rfc = RandomForestClassifier()
rfc.fit(X,Y)
#
volume = int(input('EAU: '))
smc = int(float(input('SMC: '))*100)
print('\nComputer Predicted Successful Margins Are: ')

for margin in range(1,86):
    price = 100*smc/(100 - margin)
    profit = (price - smc)*volume
    salesPrediction = rfc.predict([[volume, price, profit, margin]])
    if salesPrediction > 0:
        print('Margin = ' + str(margin) + '%, Price = $', str(round(price,0)/100) + ', Sales = $' + str(salesPrediction[0]))
print('End of List\n\nPast Successful Quotes')

sdpV = []
sdpS = []
sdpP = []
sdpM = []
udpV = []
udpS = []
udpP = []
udpM = []
for each in array:
    dpVolume = each[0]
    dpPrice = each[1]
    dpSmc = dpPrice - (each[2]/dpVolume)*100
    dpSales = each[6]
    if volume*0.8 < dpVolume < volume*1.2 and smc*0.8 < dpSmc < smc*1.2:
        if dpSales == 0:
            udpV += [dpVolume]
            udpS += [dpSmc/100]
            udpP += [dpPrice/100]
            udpM += [(1 - (dpSmc/dpPrice))*100]
        if dpSales > 0:
            sdpV += [dpVolume]
            sdpS += [dpSmc/100]
            sdpP += [dpPrice/100]
            sdpM += [(1 - (dpSmc/dpPrice))*100]

plt.plot(udpV,udpM,'ro',sdpV,sdpM,'go')
plt.title('Similar Quotes')
plt.xlabel('Volume')
plt.ylabel('Margin')

for x in range(0,len(sdpV)):
    print('Volume = '+str(round(sdpV[x],0))+', SMC = $'+str(round(sdpS[x],2))+
          ', Price = $'+str(sdpP[x])+', Margin = '+
          str( round( 100*( 1-(sdpS[x]/sdpP[x]) ),0 ) ) +'%')
print('End of List\n')

print('Unsuccessful Quotes:')
for x in range(0,len(udpV)):
    print('Volume = '+str(round(udpV[x],0))+', SMC = $'+str(round(udpS[x],2))+
          ', Price = $'+str(udpP[x])+', Margin = '+
          str( round( 100*( 1-(udpS[x]/udpP[x]) ),0 ) ) +'%')
print('End of List')

print('\nHistorical Win Ratio: '+str( round( len(sdpV)/(len(sdpV)+len(udpV))*100,0 ) )+'%')