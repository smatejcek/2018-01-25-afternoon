import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


df = pd.read_csv("adult.data.txt", skipinitialspace = True)
df.columns = ["age",
"workclass",
"fnlwgt",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"capital-gain",
"capital-loss",
"hours-per-week",
"native-country",
"income"]

df_test = pd.read_csv("adult.test.txt", skipinitialspace = True, skiprows = [0])
df_test.columns = ["age",
"workclass",
"fnlwgt",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"capital-gain",
"capital-loss",
"hours-per-week",
"native-country",
"income"]

#df["moreThanFifty"] = df["income"].apply(lambda x: 1 if x == " >50K" else 0)
df = pd.get_dummies(df)
df_test = pd.get_dummies(df)

y = df["income_>50K"]
X = df.loc[:, :"income_<=50K"]

#X = df[["age",]]

y_test = df_test["income_>50K"]
X_test = df_test.loc[:, :"income_<=50K"]

model = RandomForestClassifier(n_estimators = 100, n_jobs = 20).fit(X, y)
print(model.score(X_test, y_test))
# 100% When testing against the test data set
# That means I'm great at this, right?

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier(n_estimators = 100, n_jobs = 20).fit(X_train, y_train)
print(model.score(X_test, y_test))
# Another 100%
# Nailed it!
