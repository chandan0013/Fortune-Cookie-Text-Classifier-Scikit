#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Question 1  Naive Bayes Classifier

import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

with open('C:/Users/Chandan/Documents/CPTS 570/Homework/HW 3/fortunecookiedata/traindata.txt', 'r') as f:
    Train_x = f.readlines()
with open('C:/Users/Chandan/Documents/CPTS 570/Homework/HW 3/fortunecookiedata/trainlabels.txt', 'r') as f:
    Train_y = f.readlines()
with open('C:/Users/Chandan/Documents/CPTS 570/Homework/HW 3/fortunecookiedata/stoplist.txt', 'r') as f:
    Stop = f.readlines() 

ss = []
for i in range(len(Stop)):
    ss.append(Stop[i].rstrip()) #Striping the \n character
    
voc = []
for i in range(len(Train_x)):
    x = Train_x[i]
    for word in x.split():
        if word not in ss:
            voc.append(word)        #creating the vocabulary
    
voc.sort()
voc_sort = list(dict.fromkeys(voc))    #alphabetically sorted unique vocabulary


# Feature matrix of Training samples
X_train = np.zeros((len(Train_x), len(voc_sort)))
for i in range(len(Train_x)):
    x = Train_x[i]
    for j in range(len(voc_sort)):
        if x.find(voc_sort[j]) + 1:           
            X_train[(i,j)] =  1

Y_train = []
for i in range(len(Train_y)):
    Y_train.append(Train_y[i].rstrip()) #Striping the \n character
      
# Feature matrix of Test samples
with open('C:/Users/Chandan/Documents/CPTS 570/Homework/HW 3/fortunecookiedata/testdata.txt', 'r') as f:
    Test_x = f.readlines()
with open('C:/Users/Chandan/Documents/CPTS 570/Homework/HW 3/fortunecookiedata/testlabels.txt', 'r') as f:
    Test_y = f.readlines()
    
        

X_test = np.zeros((len(Test_x), len(voc_sort)))
for i in range(len(Test_x)):
    x = Test_x[i]
    for j in range(len(voc_sort)):
        if x.find(voc_sort[j]) + 1:           
            X_test[(i,j)] =  1   

    
Y_test = []
for i in range(len(Test_y)):
    Y_test.append(Test_y[i].rstrip()) #Striping the \n character
    
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
    
##################################################Scikit Function######################################   
    
gnb = GaussianNB()
GnBTraining_acc = []
GnBTest_acc = []


count = 0
for i in range(len(X_train)):
    y = gnb.fit(X_train,Y_train).predict([X_train[i,:]])                  # Laplace Smoothing by default
    if y != Y_train[i]:
        count = count + 1 
GnBTraining_acc.append((len(X_train)-count)/len(X_train)*100)

    
count = 0
for i in range(len(X_test)):
    y = gnb.fit(X_train,Y_train).predict([X_test[i,:]])                   # Laplace Smoothing by default
    if y != Y_test[i]:
        count = count + 1 
GnBTest_acc.append((len(X_test)-count)/len(X_test)*100)

print('Scikit GnB Training accuracy is:', GnBTraining_acc,'%' )
print('Scikit GnB Test accuracy is:', GnBTest_acc ,'%' )

#############################################Defining Naive Bayes Classifier###################################

c1 = 0
for i in range(len(Y_train)):
    if int(Y_train[i]) == 1:
        c1 = c1 + 1
P1 = c1/len(Y_train)

Training_acc = []
Test_acc = []

error = 0
for i in range(len(X_train)):
    x = X_train[i,:]
    mult1 = 1
    mult0 = 1
    for ii in range(len(voc_sort)):
        count = 0
        for iii in range(len(Y_train)):
            if int(Y_train[iii]) == 1:
                if X_train[iii,ii] == x[ii]:
                    count = count + 1
        # for all y = 1, #count times input feature ii takes that value
        Px_i1 = (count/c1)          
        mult1 = mult1*Px_i1               # independent events
        mult0 = mult0*(1 - Px_i1)
            
            
        
    P1 = ((P1*mult1)+1)/(((1-P1)*mult0) + (P1*mult1)+2) #Bayes rule with Laplace smooting
    P0 = 1 - P1
    y_pred = 0
    if P1 > P0:
        y_pred = 1
    if y_pred != int(Y_train[i]):
        error = error + 1
        
Training_acc.append((len(X_train)-error)/len(X_train)*100)

error = 0
for i in range(len(X_test)):
    x = X_test[i,:]
    mult1 = 1
    mult0 = 1
    for ii in range(len(voc_sort)):
        count = 0
        for iii in range(len(Y_train)):
            if int(Y_train[iii]) == 1:
                if X_train[iii,ii] == x[ii]:
                    count = count + 1
        # for all y = 1, #count times input feature ii takes that value
        Px_i1 = (count/c1)          
        mult1 = mult1*Px_i1                   # Independent events
        mult0 = mult0*(1 - Px_i1)
            
            
        
    P1 = ((P1*mult1 + 1))/(((1-P1)*mult0) + (P1*mult1)+2) #Bayes rule with Laplace smoothing
    P0 = 1 - P1
    y_pred = 0
    if P1 > P0:
        y_pred = 1
    if y_pred != int(Y_test[i]):
        error = error + 1
        
Test_acc.append((len(X_test)-error)/len(X_test)*100)

print('Training accuracy is:', Training_acc,'%' )
print('Test accuracy is:', Test_acc ,'%' )

