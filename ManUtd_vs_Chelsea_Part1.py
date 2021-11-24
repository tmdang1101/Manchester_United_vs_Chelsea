import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier






##### Load dataset 

df_train = pd.read_csv("C:/Users/tuand/Downloads/EPL 2018_2019 - Result - 2018-19.csv")
df_test = pd.read_csv("C:/Users/tuand/Downloads/EPL 2019_2020 - Result - 2019-20.csv")
#print(df_test.head())




##### Split dataset into training and testing sets

#X = df[['Home Team','Away Team']]
#y = df[['HT Goals']]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train = df_train[['Home Team','Away Team','HT Avg Scored Goals','HT Avg Conceded Goals','AT Avg Scored Goals','AT Avg Conceded Goals']]
y_train = df_train[['Match Result']]
X_test = df_test[['Home Team','Away Team','HT Avg Scored Goals','HT Avg Conceded Goals','AT Avg Scored Goals','AT Avg Conceded Goals']]
y_test = df_test[['Match Result']]




##### Create Model
regr = DecisionTreeClassifier()

  
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))



##### Make Prediction
print(regr.predict([[1,6,27/11,4/11,19/11,17/11]]))



preds = []

for i in range(100):
    regr = DecisionTreeClassifier()
    regr.fit(X_train, y_train)
    #preds.append(regr.predict([[1,6,27/11,4/11,19/11,17/11]])[0])
    preds.append(regr.score(X_test, y_test))

plt.hist(preds)
plt.show()

print(preds)

























































































