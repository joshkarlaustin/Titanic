# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:28:40 2020

@author: Josh
"""

import pandas as pd
import numpy as np
import statistics as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import statsmodels.discrete as smd 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Load Data
titanic_total = pd.read_csv("C:\\Data\\Titanic\\titanic.csv")

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(titanic_total, 0.2)

y_train = train_set["Survived"]
y_test = test_set["Survived"]


# Setup X data (come back for age is missing)

passenger_class_train = pd.get_dummies(train_set["Pclass"], prefix="Class")
passenger_class_test = pd.get_dummies(test_set["Pclass"], prefix="Class")
gender_train = pd.get_dummies(train_set["Sex"])
gender_test = pd.get_dummies(test_set["Sex"])

meanage = st.mean(train_set["Age"].dropna())
age2_train = train_set["Age"].fillna(meanage)
age2_test = test_set["Age"].fillna(meanage)
agemiss_train = pd.isna(train_set["Age"]) * 1
agemiss_test = pd.isna(test_set["Age"]) * 1
oneSibSp_train = (train_set["SibSp"] == 1) * 1
mulSibSp_train = (train_set["SibSp"] > 1) * 1
oneSibSp_test = (test_set["SibSp"] == 1) * 1
mulSibSp_test = (test_set["SibSp"] > 1) * 1

X_first_train = pd.concat([age2_train, train_set["Parch"], train_set["Fare"], passenger_class_train["Class_1"], passenger_class_train["Class_2"], gender_train["female"], agemiss_train, oneSibSp_train, mulSibSp_train], axis=1)
#X_sec= X_first.dropna(subset=["SibSp"])
X_first_test = pd.concat([age2_test, test_set["Parch"], test_set["Fare"], passenger_class_test["Class_1"], passenger_class_test["Class_2"], gender_test["female"], agemiss_test, oneSibSp_test, mulSibSp_test], axis=1)






poly = PolynomialFeatures(2, include_bias=False)
X_poly1_train = X_first_train
X_polydraft_train = poly.fit_transform(X_first_train)
X_poly1_test = X_first_test
X_polydraft_test = poly.fit_transform(X_first_test)







def remove_redundant(X_train, X_test, thresh=100):
    cols = X_train.columns
    variables = np.arange(X_train.shape[1])
    variable_list = np.arange(X_train.shape[1] - 1, stop = -1, step = -1)
#    print(variable_list)

    for ix in variable_list:
#        print(ix)
        c = X_train[cols[variables]].values
        vif = variance_inflation_factor(c, ix)
#        print(vif)
 
        if vif > thresh or np.isnan(vif):
            print('dropping \'' + str(ix) + '\' at index: ' + str(ix))
            variables = np.delete(variables, ix)
    
    print('Remaining variables:')
    print(X_train.columns[variables])
    return X_train[cols[variables]], X_test[cols[variables]]





X_poly2_train, X_poly2_test = remove_redundant(pd.DataFrame(X_polydraft_train), pd.DataFrame(X_polydraft_test))






np.sum(X_poly2_train, axis = 0)











scaler1 = StandardScaler()
scaler1.fit(X_poly1_train)
X_scale1_train = scaler1.transform(X_poly1_train)
X_scale1_test = scaler1.transform(X_poly1_test)
scaler2 = StandardScaler()
scaler2.fit(X_poly2_train)
X_scale2_train = scaler2.transform(X_poly2_train)
X_scale2_test = scaler2.transform(X_poly2_test)

pd.DataFrame(X_scale2_train).hist(bins=20, figsize = (20,15))











    

X_array1 = np.array(X_scale1_train)
X_prime_X1 = np.transpose(X_array1) @ X_array1 / 713
np.linalg.matrix_rank(X_prime_X1)
X_inv1 = np.linalg.inv(X_prime_X1)
diag1 = 1 / np.sqrt(np.diag(X_inv1))
recenter = np.mean(diag1)
multiplier1 = np.diag(diag1) / recenter
X_rescale1a_train = X_scale1_train @ multiplier1
X_rescale1a_test = X_scale1_test @ multiplier1

X_array2 = np.array(X_scale2_train)
X_prime_X2 = np.transpose(X_array2) @ X_array2 / 713
print(np.linalg.matrix_rank(X_prime_X2))
X_inv2 = np.linalg.inv(X_prime_X2)
diag2 = 1 / np.sqrt(np.diag(X_inv2))
recenter = np.mean(diag2)
multiplier2 = np.diag(diag2) / recenter
X_rescale2a_train = X_scale2_train @ multiplier2
X_rescale2a_test = X_scale2_test @ multiplier2

XpX2inspect = pd.DataFrame(X_prime_X2)
corrMat = XpX2inspect.corr()




# Run logit in statsmodel
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_scale1_train)).fit()
diag1 = np.sqrt(np.diag(np.array(model.normalized_cov_params)))
recenter = np.mean(diag1)
multiplier1 = np.diag(diag1[1:]) / recenter
X_rescale1b_train = X_scale1_train @ multiplier1
X_rescale1b_test = X_scale1_test @ multiplier1


# Run logit in statsmodel
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_rescale1b_train)).fit()
print_coef = model.summary()
print(print_coef)
margeff = model.get_margeff()
print_margeff = margeff.summary()
print(print_margeff)




# Run logit in statsmodel
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_scale2_train)).fit()
print(model.summary())
diag2 = np.sqrt(np.diag(np.array(model.normalized_cov_params)))
recenter = np.mean(diag2)
multiplier2 = np.diag(diag2[1:]) / recenter
X_rescale2b_train = X_scale2_train @ multiplier2
X_rescale2b_test = X_scale2_test @ multiplier2



# Run logit in statsmodel
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_rescale2b_train)).fit()
print_coef = model.summary()
print(print_coef)
margeff = model.get_margeff()
print_margeff = margeff.summary()
print(print_margeff)


'''
# Run logit in statsmodel
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_rescale2b)).fit()
print_coef = model.summary()
print(print_coef)
margeff = model.get_margeff()
print_margeff = margeff.summary()
print(print_margeff)
statpred = model.predict(sm.add_constant(X_rescale2b))
'''






'''
logitreg2z = LogisticRegression().fit(X_scale2_train, y_train)
logitreg2z.coef_
logpred2z = logitreg2z.predict(X_scale2_train)
logodds2z = logitreg2z.predict_proba(X_scale2)
sum(y_train)
sum(logpred2z)
sum(logodds2z)
print(confusion_matrix(y_train, logpred2z))


logitreg2aB = LogisticRegression().fit(X_rescale2a, y_train)
print(logitreg2a.coef_)
print(logitreg2a.intercept_)
logpred2a = logitreg2a.predict(X_rescale2a)
logodds2a = logitreg2a.predict_proba(X_rescale2a)
sum(y_train)
sum(logpred2a)
sum(logodds2a)
print(confusion_matrix(y_train, logpred2a))


logitreg2b = LogisticRegression().fit(X_rescale2b, y_train)
print(logitreg2b.coef_)
print(logitreg2b.intercept_)
logpred2b = logitreg2b.predict(X_rescale2b)
logodds2b = logitreg2b.predict_proba(X_rescale2b)
print(sum(y_train))
print(sum(logpred2b))
print(sum(logodds2b))
print(confusion_matrix(y_train, logpred2b))

betterpred2b = (logodds2b[:,1] > sorted(logodds2b[:,1])[500]) * 1
confusion_matrix(y_train, betterpred2b)
sum(betterpred2b)
'''



logitreg1zE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_scale1_train, y_train)
logpred1zE = logitreg1zE.predict(X_scale1_test)
print("1zE Confusion")
print(confusion_matrix(y_test, logpred1zE))
logitreg1aE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_rescale1a_train, y_train)
logpred1aE = logitreg1aE.predict(X_rescale1a_test)
print("1aE Confusion")
print(confusion_matrix(y_test, logpred1aE))
logitreg1bE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_rescale1b_train, y_train)
logpred1bE = logitreg1bE.predict(X_rescale1b_test)
print("1bE Confusion")
print(confusion_matrix(y_test, logpred1bE))







logitreg2zL = LogisticRegression(penalty = "l1", solver = "liblinear", intercept_scaling = 1000000).fit(X_scale2_train, y_train)
logpred2zL = logitreg2zL.predict(X_scale2_train)
confusion_matrix(y_train, logpred2zL)

logitreg2aL = LogisticRegression(penalty = "l1", solver = "liblinear", intercept_scaling = 1000000).fit(X_rescale2a_train, y_train)
logpred2aL = logitreg2aL.predict(X_rescale2a_train)
confusion_matrix(y_train, logpred2aL)

logitreg2bL = LogisticRegression(penalty = "l1", solver = "liblinear", intercept_scaling = 1000000).fit(X_rescale2b_train, y_train)
logpred2bL = logitreg2bL.predict(X_rescale2b_train)
confusion_matrix(y_train, logpred2bL)



logitreg2zE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_scale2_train, y_train)
logpred2zE_train = logitreg2zE.predict(X_scale2_train)
logpred2zE_test = logitreg2zE.predict(X_scale2_test)
# outpred2zE = logitreg2zE.predict(X_scale2)
print("2zE Confusion")
print(confusion_matrix(y_test, logpred2zE_test))

logitreg2aE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_rescale2a_train, y_train)
logpred2aE_train = logitreg2aE.predict(X_rescale2a_train)
logpred2aE_test = logitreg2aE.predict(X_rescale2a_test)
print("2aE Confusion")
print(confusion_matrix(y_test, logpred2aE_test))

logitreg2bE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_rescale2b_train, y_train)
logpred2bE_train = logitreg2bE.predict(X_rescale2b_train)
logpred2bE_test = logitreg2bE.predict(X_rescale2b_test)
print("2bE Confusion")
print(confusion_matrix(y_test, logpred2bE_test))






print(logitreg2zE.coef_)
print(logitreg2aE.coef_)
print(logitreg2bE.coef_)









