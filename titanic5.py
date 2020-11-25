# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:28:40 2020

@author: Josh

This is a draft at an attempt to predict survivors from the Titanic as described in the relevant compettiton at Kaggle.
It incorporates a draft of a method to rescale variables unevenly to weight variables with lower multicollinearity higher.
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import statsmodels.discrete as smd 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.cluster import KMeans



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
    passenger_class = pd.get_dummies(data["Pclass"], prefix="Class")
    gender = pd.get_dummies(data["Sex"])
    
    meanage = st.mean(data["Age"].dropna())
    age2 = data["Age"].fillna(meanage)
    agemiss = pd.isna(data["Age"]) * 1
    agemiss = agemiss.rename("AgeMiss")
    oneSibSp = (data["SibSp"] == 1) * 1
    oneSibSp = oneSibSp.rename("withOne")
    mulSibSp = (data["SibSp"] > 1) * 1
    mulSibSp = mulSibSp.rename("withMany")
 
    name = data["Name"]
    pattern = r', ([A-Za-z]+).'
    salutation = name.str.extract(pattern)
    sal_Mr = pd.DataFrame((salutation == 'Mr') * 1)
    sal_Mr = sal_Mr.rename(columns = {0 : "Mr"})
    sal_Off = pd.DataFrame(((salutation == "Capt") | (salutation == 'Major') | (salutation == 'Col')) * 1)
    sal_Off = sal_Off.rename(columns = {0:"Officer"})
    sal_Dr = pd.DataFrame((salutation == "Dr") * 1)
    sal_Dr = sal_Dr.rename(columns = {0:"Dr"})
    sal_Prt = pd.DataFrame(((salutation == "Rev") | (salutation == 'Don')) * 1)
    sal_Prt = sal_Prt.rename(columns = {0:"Priest"})
    sal_Mas = pd.DataFrame(((salutation == "Master") | (salutation == "Sir") | (salutation == "Jonkheer")) * 1)
    sal_Mas = sal_Mas.rename(columns = {0:"Master"})
    sal_Miss = pd.DataFrame(((salutation == "Miss") | (salutation == "Ms") | (salutation == "Mlle")) * 1)
    sal_Miss = sal_Miss.rename(columns = {0:"Miss"})
    sal_Mrs = pd.DataFrame(((salutation == "Mrs") | (salutation == "Mme") | (salutation == "Lady")) * 1)
    sal_Mrs = sal_Mrs.rename(columns = {0:"Mrs"})
    salutation_dummies = pd.concat([sal_Off, sal_Dr, sal_Prt, sal_Mas, sal_Miss, sal_Mrs], axis=1)
       
    # Build combined X datasets from raw variables and dummy variables (no scaling or polynomial)
    X_first = pd.concat([age2, data["Parch"], data["Fare"], passenger_class["Class_1"], passenger_class["Class_3"], salutation_dummies, agemiss, oneSibSp, mulSibSp], axis=1)
    return X_first

X_first_train = create_titanic_features(train_set)
X_first_test = create_titanic_features(test_set)
    

# Create Polynomial features for the data (still no scaling)
poly = PolynomialFeatures(2, include_bias=False)
X_polydraft1_train = poly.fit_transform(X_first_train)
X_polydraft2_train = pd.DataFrame(X_polydraft1_train, columns = poly.get_feature_names(X_first_train.columns))
X_polydraft1_test = poly.fit_transform(X_first_test)
X_polydraft2_test = pd.DataFrame(X_polydraft1_test, columns = poly.get_feature_names(X_first_test.columns))


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
    return X_train[cols[variables]], X_test[cols[variables]], X_train.columns[variables]

X_poly2_train, X_poly2_test, remaining_variables2 = remove_redundant(pd.DataFrame(X_polydraft2_train), pd.DataFrame(X_polydraft2_test), thresh = 100)
X_poly1_train, X_poly1_test, remaining_variables1 = remove_redundant(pd.DataFrame(X_first_train), pd.DataFrame(X_first_test), thresh = 100)


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
def homebrew_rescale1(train_data, test_data):
    X_array = np.array(train_data)
    X_prime_X = np.transpose(X_array) @ X_array / len(X_array)
    np.linalg.matrix_rank(X_prime_X)
    X_inv = np.linalg.inv(X_prime_X)
    diag = 1 / np.sqrt(np.diag(X_inv))
    recenter = np.mean(diag)
    multiplier = np.diag(diag) / recenter
    X_rescale_train = train_data @ multiplier
    X_rescale_test = test_data @ multiplier
    return X_rescale_train, X_rescale_test

X_rescale1a_train, X_rescale1a_test = homebrew_rescale1(X_scale1_train, X_scale1_test)
X_rescale2a_train, X_rescale2a_test = homebrew_rescale1(X_scale2_train, X_scale2_test)



# Run "homebrew" scaling method 2 such that all logit standard errors are the same
def homebrew_rescale2(train_data, test_data):
    model = smd.discrete_model.Logit(y_train, sm.add_constant(train_data)).fit(method='bfgs', maxiter=250)
    print_coef = model.summary()
    print(print_coef)
    diagA = np.sqrt(np.diag(np.array(model.normalized_cov_params)))
    diagB = (diagA * (diagA < 10)) + (0.5 * (min(diagA)) * (diagA > 10))
    recenter = np.mean(diagB[1:])
    multiplier = np.diag(diagB[1:]) / recenter
    X_rescale_train = train_data @ multiplier
    X_rescale_test = test_data @ multiplier
    return X_rescale_train, X_rescale_test

X_rescale1b_train, X_rescale1b_test = homebrew_rescale2(X_scale1_train, X_scale1_test)
X_rescale2b_train, X_rescale2b_test = homebrew_rescale2(X_scale2_train, X_scale2_test)






clf = DecisionTreeClassifier()
model = clf.fit(X_scale1_train, y_train)
treepred = clf.predict(X_scale1_test)
print(metrics.confusion_matrix(y_test, treepred))
print(metrics.f1_score(y_test, treepred))





clf = DecisionTreeClassifier()
model = clf.fit(X_scale2_train, y_train)
treepred = clf.predict(X_scale2_test)
print(metrics.confusion_matrix(y_test, treepred))
print(metrics.f1_score(y_test, treepred))









kmeans = KMeans(n_clusters = 10)
kmeanmodel = kmeans.fit(X_scale1_train, y_train)
clusterpredtrain = kmeans.predict(X_scale1_train)
clusterpredtest = kmeans.predict(X_scale1_test)
onehotfit = OneHotEncoder().fit(clusterpredtrain.reshape(-1,1))
clusterOneHottrain = onehotfit.transform(clusterpredtrain.reshape(-1,1))
clusterOneHottest = onehotfit.transform(clusterpredtest.reshape(-1,1))
centers = np.transpose(kmeans.cluster_centers_)
centerDF = pd.DataFrame(centers ,remaining_variables1)
clusterlogit = LogisticRegression().fit(clusterOneHottrain, y_train)
predtrain = clusterlogit.predict(clusterOneHottrain)
print(metrics.confusion_matrix(y_train, predtrain))
print(metrics.f1_score(y_train, predtrain))
predtest = clusterlogit.predict(clusterOneHottest)
print(metrics.confusion_matrix(y_test, predtest))
print(metrics.f1_score(y_test, predtest))


















# Check performance of scaling methods for models without polynomial terms
logitreg1zE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_scale1_train, y_train)
logpred1zE = logitreg1zE.predict(X_scale1_test)
print("1zE Confusion")
print(metrics.confusion_matrix(y_test, logpred1zE))
print(metrics.f1_score(y_test, logpred1zE))
logitreg1aE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_rescale1a_train, y_train)
logpred1aE = logitreg1aE.predict(X_rescale1a_test)
print("1aE Confusion")
print(metrics.confusion_matrix(y_test, logpred1aE))
print(metrics.f1_score(y_test, logpred1aE))
logitreg1bE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_rescale1b_train, y_train)
logpred1bE = logitreg1bE.predict(X_rescale1b_test)
print("1bE Confusion")
print(metrics.confusion_matrix(y_test, logpred1bE))
print(metrics.f1_score(y_test, logpred1bE))




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
print(metrics.confusion_matrix(y_test, logpred2zE_test))
print(metrics.f1_score(y_test, logpred2zE_test))

logitreg2aE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_rescale2a_train, y_train)
logpred2aE_train = logitreg2aE.predict(X_rescale2a_train)
logpred2aE_test = logitreg2aE.predict(X_rescale2a_test)
print("2aE Confusion")
print(metrics.confusion_matrix(y_test, logpred2aE_test))
print(metrics.f1_score(y_test, logpred2aE_test))

logitreg2bE = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.5, max_iter = 100000, tol=1e-8).fit(X_rescale2b_train, y_train)
logpred2bE_train = logitreg2bE.predict(X_rescale2b_train)
logpred2bE_test = logitreg2bE.predict(X_rescale2b_test)
print("2bE Confusion")
print(metrics.confusion_matrix(y_test, logpred2bE_test))
print(metrics.f1_score(y_test, logpred2bE_test))

'''
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
'''





logit1ze_coef = pd.DataFrame(np.transpose(logitreg1zE.coef_),remaining_variables1)
logit2ze_coef = pd.DataFrame(np.transpose(logitreg2zE.coef_),remaining_variables2)

print(logitreg2aE.coef_)
print(logitreg2bE.coef_)









