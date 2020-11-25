# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:28:40 2020

@author: Josh

This is a draft at an attempt to predict survivors from the Titanic as described in the relevant compettiton at Kaggle.
It incorporates a draft of a method to rescale variables unevenly to weight variables with lower multicolinearity higher.
This will cause those to be penalized less by regularization methods like ridge and lasso and emphasize the "better" 
(less multicolinear) variables when making predictions.

"""

# Load libraries
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


# Load Data and split it
titanic_total = pd.read_csv("C:\\Data\\Titanic\\titanic.csv")

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(titanic_total, 0.2)


# Set outcome variables
y_train = train_set["Survived"]
y_test = test_set["Survived"]


# Create dummy variables
def create_titanic_features(data):
passenger_class_train = pd.get_dummies(train_set["Pclass"], prefix="Class")
passenger_class_test = pd.get_dummies(test_set["Pclass"], prefix="Class")
gender_train = pd.get_dummies(train_set["Sex"])
gender_test = pd.get_dummies(test_set["Sex"])

meanage = st.mean(train_set["Age"].dropna())
age2_train = train_set["Age"].fillna(meanage)
age2_test = test_set["Age"].fillna(meanage)
agemiss_train = pd.isna(train_set["Age"]) * 1
agemiss_train = agemiss_train.rename("AgeMiss")
agemiss_test = pd.isna(test_set["Age"]) * 1
agemiss_test = agemiss_test.rename("AgeMiss")
oneSibSp_train = (train_set["SibSp"] == 1) * 1
oneSibSp_train = oneSibSp_train.rename("withOne")
mulSibSp_train = (train_set["SibSp"] > 1) * 1
mulSibSp_train = mulSibSp_train.rename("withMany")
oneSibSp_test = (test_set["SibSp"] == 1) * 1
oneSibSp_test = oneSibSp_test.rename("withOne")
mulSibSp_test = (test_set["SibSp"] > 1) * 1
mulSibSp_test = mulSibSp_test.rename("withMany")

name = titanic_total["Name"]
pattern = r', ([A-Za-z]+).'
salutation = name.str.extract(pattern)
salutation.value_counts()

name = train_set["Name"]
pattern = r', ([A-Za-z]+).'
salutation = name.str.extract(pattern)
sal_Mr_train = pd.DataFrame((salutation == 'Mr') * 1)
sal_Mr_train = sal_Mr_train.rename(columns = {0 : "Mr"})
sal_Off_train = pd.DataFrame(((salutation == "Capt") | (salutation == 'Major') | (salutation == 'Col')) * 1)
sal_Off_train = sal_Off_train.rename(columns = {0:"Officer"})
sal_Dr_train = pd.DataFrame((salutation == "Dr") * 1)
sal_Dr_train = sal_Dr_train.rename(columns = {0:"Dr"})
sal_Prt_train = pd.DataFrame(((salutation == "Rev") | (salutation == 'Don')) * 1)
sal_Prt_train = sal_Prt_train.rename(columns = {0:"Priest"})
sal_Mas_train = pd.DataFrame(((salutation == "Master") | (salutation == "Sir") | (salutation == "Jonkheer")) * 1)
sal_Mas_train = sal_Mas_train.rename(columns = {0:"Master"})
sal_Miss_train = pd.DataFrame(((salutation == "Miss") | (salutation == "Ms") | (salutation == "Mlle")) * 1)
sal_Miss_train = sal_Miss_train.rename(columns = {0:"Miss"})
sal_Mrs_train = pd.DataFrame(((salutation == "Mrs") | (salutation == "Mme") | (salutation == "Lady")) * 1)
sal_Mrs_train = sal_Mrs_train.rename(columns = {0:"Mrs"})
salutation_dummies_train = pd.concat([sal_Off_train, sal_Dr_train, sal_Prt_train, sal_Mas_train, sal_Miss_train, sal_Mrs_train], axis=1)


name = test_set["Name"]
pattern = r', ([A-Za-z]+).'
salutation = name.str.extract(pattern)
sal_Mr_test = pd.DataFrame((salutation == 'Mr') * 1)
sal_Mr_test = sal_Mr_test.rename(columns = {0 : "Mr"})
sal_Off_test = pd.DataFrame(((salutation == "Capt") | (salutation == 'Major') | (salutation == 'Col')) * 1)
sal_Off_test = sal_Off_test.rename(columns = {0:"Officer"})
sal_Dr_test = pd.DataFrame((salutation == "Dr") * 1)
sal_Dr_test = sal_Dr_test.rename(columns = {0:"Dr"})
sal_Prt_test = pd.DataFrame(((salutation == "Rev") | (salutation == 'Don')) * 1)
sal_Prt_test = sal_Prt_test.rename(columns = {0:"Priest"})
sal_Mas_test = pd.DataFrame(((salutation == "Master") | (salutation == "Sir") | (salutation == "Jonkheer")) * 1)
sal_Mas_test = sal_Mas_test.rename(columns = {0:"Master"})
sal_Miss_test = pd.DataFrame(((salutation == "Miss") | (salutation == "Ms") | (salutation == "Mlle")) * 1)
sal_Miss_test = sal_Miss_test.rename(columns = {0:"Miss"})
sal_Mrs_test = pd.DataFrame(((salutation == "Mrs") | (salutation == "Mme") | (salutation == "Lady")) * 1)
sal_Mrs_test = sal_Mrs_test.rename(columns = {0:"Mrs"})
salutation_dummies_test = pd.concat([sal_Off_test, sal_Dr_test, sal_Prt_test, sal_Mas_test, sal_Miss_test, sal_Mrs_test], axis=1)


# Build combined X datasets from raw variables and dummy variables (no scaling or polynomial)
X_first_train = pd.concat([age2_train, train_set["Parch"], train_set["Fare"], passenger_class_train["Class_1"], passenger_class_train["Class_2"], salutation_dummies_train, agemiss_train, oneSibSp_train, mulSibSp_train], axis=1)
#X_sec= X_first.dropna(subset=["SibSp"])
X_first_test = pd.concat([age2_test, test_set["Parch"], test_set["Fare"], passenger_class_test["Class_1"], passenger_class_test["Class_2"], salutation_dummies_test, agemiss_test, oneSibSp_test, mulSibSp_test], axis=1)


# Create Polynomial features for the data (still no scaling)
poly = PolynomialFeatures(2, include_bias=False)
X_polydraft_train = poly.fit_transform(X_first_train)
X_polydraft_test = poly.fit_transform(X_first_test)


# Remove extremely multicolinear and redundant variables
def remove_redundant(X_train, X_test, thresh=100):
    cols = X_train.columns
    variables = np.arange(X_train.shape[1])
    variable_list = np.arange(X_train.shape[1] - 1, stop = -1, step = -1)
    print(variable_list)

    for ix in variable_list:
        print(ix)
        c = X_train[cols[variables]].values
        vif = variance_inflation_factor(c, ix)
        print(vif)
 
        if vif > thresh or np.isnan(vif):
            print('dropping \'' + str(ix) + '\' at index: ' + str(ix))
            variables = np.delete(variables, ix)
    
    print('Remaining variables:')
    print(X_train.columns[variables])
    return X_train[cols[variables]], X_test[cols[variables]]

X_poly2_train, X_poly2_test = remove_redundant(pd.DataFrame(X_polydraft_train), pd.DataFrame(X_polydraft_test), thresh = 10)
X_poly1_train, X_poly1_test = remove_redundant(pd.DataFrame(X_first_train), pd.DataFrame(X_first_test), thresh = 10)


# Run standard scaling methods
scaler1 = StandardScaler()
scaler1.fit(X_poly1_train)
X_scale1_train = scaler1.transform(X_poly1_train)
X_scale1_test = scaler1.transform(X_poly1_test)
scaler2 = StandardScaler()
scaler2.fit(X_poly2_train)
X_scale2_train = scaler2.transform(X_poly2_train)
X_scale2_test = scaler2.transform(X_poly2_test)

#pd.DataFrame(X_scale2_train).hist(bins=20, figsize = (20,15))



# Run "homebrew" scaling method 1 such that all OLS standard errors are the same
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




# Run "homebrew" scaling method 2 such that all logit standard errors are the same
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_scale1_train)).fit(method='bfgs', maxiter=250)
print_coef = model.summary()
print(print_coef)
diag1a = np.sqrt(np.diag(np.array(model.normalized_cov_params)))
diag1b = (diag1a * (diag1a < 10)) + (0.5 * (min(diag1a)) * (diag1a > 10))
recenter = np.mean(diag1b)
multiplier1 = np.diag(diag1b[1:]) / recenter
X_rescale1b_train = X_scale1_train @ multiplier1
X_rescale1b_test = X_scale1_test @ multiplier1

'''
# Verify homebrew scaling method worked 
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_rescale1b_train)).fit(method='bfgs', maxiter=250)
print_coef = model.summary()
print(print_coef)
margeff = model.get_margeff()
print_margeff = margeff.summary()
print(print_margeff)
'''

# Run method 2 for 2nd order polynomial
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_scale2_train)).fit(method='bfgs', maxiter=250)
print(model.summary())
diag2a = np.sqrt(np.diag(np.array(model.normalized_cov_params)))
diag2b = (diag2a * (diag2a < 10)) + (0.1 * (min(diag2a)) * (diag2a > 10))
recenter = np.mean(diag2b)
multiplier2 = np.diag(diag2b[1:]) / recenter
X_rescale2b_train = X_scale2_train @ multiplier2
X_rescale2b_test = X_scale2_test @ multiplier2

'''
# Verify homebrew method worked
model = smd.discrete_model.Logit(y_train, sm.add_constant(X_rescale2b_train)).fit(method='bfgs', maxiter=250)
print_coef = model.summary()
print(print_coef)
margeff = model.get_margeff()
print_margeff = margeff.summary()
print(print_margeff)
'''





# Check performance of scaling methods for models without polynomial terms
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





"""
# Check performance of models with Lasso
logitreg2zL = LogisticRegression(penalty = "l1", solver = "liblinear", intercept_scaling = 1000000).fit(X_scale2_train, y_train)
logpred2zL = logitreg2zL.predict(X_scale2_train)
confusion_matrix(y_train, logpred2zL)

logitreg2aL = LogisticRegression(penalty = "l1", solver = "liblinear", intercept_scaling = 1000000).fit(X_rescale2a_train, y_train)
logpred2aL = logitreg2aL.predict(X_rescale2a_train)
confusion_matrix(y_train, logpred2aL)

logitreg2bL = LogisticRegression(penalty = "l1", solver = "liblinear", intercept_scaling = 1000000).fit(X_rescale2b_train, y_train)
logpred2bL = logitreg2bL.predict(X_rescale2b_train)
confusion_matrix(y_train, logpred2bL)
"""


# Check performance of models with polynomial terms using elastic net models
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

logitreg2bE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8, C = 0.1).fit(X_rescale2b_train, y_train)
logpred2bE_train = logitreg2bE.predict(X_rescale2b_train)
logpred2bE_test = logitreg2bE.predict(X_rescale2b_test)
print("2bE Confusion (low penalty)")
print(confusion_matrix(y_test, logpred2bE_test))

logitreg2bE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8, C = 10).fit(X_rescale2b_train, y_train)
logpred2bE_train = logitreg2bE.predict(X_rescale2b_train)
logpred2bE_test = logitreg2bE.predict(X_rescale2b_test)
print("2bE Confusion (high penalty)")
print(confusion_matrix(y_test, logpred2bE_test))






print(logitreg2zE.coef_)
print(logitreg2aE.coef_)
print(logitreg2bE.coef_)









