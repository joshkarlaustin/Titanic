# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:15:46 2020

@author: Josh
"""


import pandas as pd
import numpy as np
#import sklearn.datasets as md
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


betE = 1
bet2 = 1
bet3 = 1
bet4 = 1
bet5 = -1
bet6 = 1
bet7 = 1

muE = 0
mu2 = 0
mu3 = 0
mu4 = 0
mu5 = 0
mu6 = 0
mu7 = 0

sdE = 3
sd2 = 1
sd3 = 1
sd4 = 1
sd5 = 1
sd6 = 1
sd7 = 1

bias2 = 0
bias3 = 0
bias4 = 0
bias5 = 0
bias6 = 0
bias7 = 0

multi23 = 0.99
multi24 = 0
multi25 = 0
multi26 = 0
multi27 = 0
multi34 = 0
multi35 = 0
multi36 = 0
multi37 = 0
multi45 = 0.99
multi46 = 0
multi47 = 0
multi56 = 0
multi57 = 0
multi67 = 0


obs = 1000

mean = [muE, mu2, mu3, mu4, mu5, mu6, mu7]
variances = np.array([sdE, sd2, sd3, sd4, sd5, sd6, sd7])
varmat = np.outer(variances, np.transpose(variances))
cov = np.array([[1, bias2, bias3, bias4, bias5, bias6, bias7], [bias2, 1, multi23, multi24, multi25, multi26, multi27], [bias3, multi23, 1, multi34, multi35, multi36, multi37], [bias4, multi24, multi34, 1, multi45, multi46, multi47], [bias5, multi25, multi35, multi45, 1, multi56, multi57], [bias6, multi26, multi36, multi46, multi56, 1, multi67], [bias7, multi27, multi37, multi47, multi57, multi67, 1]])
varcov = varmat * cov
multiplier = [[betE], [bet2], [bet3], [bet4], [bet5], [bet6], [bet7]]


rawdata = np.random.multivariate_normal(mean, cov, obs)
latent = rawdata @ multiplier
outcome = (latent > 0) * 1
X_data = rawdata[:,1:]
combined = pd.DataFrame(np.append(outcome, X_data, axis = 1))




def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(combined, 0.2)





y_train = train_set[0]
X_train = train_set[{1,2,3,4,5,6}]
y_test = test_set[0]
X_test = test_set[{1,2,3,4,5,6}]




scaler1 = StandardScaler()
scaler1.fit(X_train)
X_scale_train = scaler1.transform(X_train)
X_scale_test = scaler1.transform(X_test)



'''
alpha = 1
X_array = np.array(X_scale_train)
X_prime_X = np.transpose(X_array) @ X_array / len(X_array)
alphaA = alpha * np.identity(4)
XpX_inv = np.linalg.inv(X_prime_X)
XpXaA_inv = np.linalg.inv(X_prime_X + alphaA)
XpY = np.transpose(X_array) @ np.array(y_train) / len(X_array)
coefNone = XpX_inv @ XpY
coefRidge = XpXaA_inv @ XpY
'''

truelatent = X_test @ multiplier[1:] 
truepred = (truelatent > 0) * 1
print(confusion_matrix(y_test, truepred))
print(f1_score(y_test, truepred))




model_none = LogisticRegression(penalty = 'none').fit(X_scale_train, y_train)
logpred = model_none.predict(X_scale_test)
print(confusion_matrix(y_test, logpred))
print(f1_score(y_test, logpred))
print(model_none.coef_)




model_l1 = LogisticRegression(penalty = 'l1', solver = 'liblinear').fit(X_scale_train, y_train)
logpred = model_l1.predict(X_scale_test)
print(confusion_matrix(y_test, logpred))
print(f1_score(y_test, logpred))
print(model_l1.coef_)




model_l2 = LogisticRegression(penalty = 'l2').fit(X_scale_train, y_train)
logpred = model_l2.predict(X_scale_test)
print(confusion_matrix(y_test, logpred))
print(f1_score(y_test, logpred))
print(model_l2.coef_)




model_EN = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5).fit(X_scale_train, y_train)
logpred = model_EN.predict(X_scale_test)
print(confusion_matrix(y_test, logpred))
print(f1_score(y_test, logpred))
print(model_EN.coef_)



