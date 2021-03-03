import pandas as pd

df = pd.read_csv('Iris.csv')

df.drop(['Id'],axis=1,inplace=True)

df.Species[df.Species == 'Iris-setosa'] = 1
df.Species[df.Species == 'Iris-versicolor'] = 2
df.Species[df.Species == 'Iris-virginica'] = 3

X=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df[['Species']]

y=y.astype(int)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LR = LogisticRegression()
LR.fit(X_train, y_train)

pred=LR.predict(X_test)
accuracy_score(pred,y_test)

import pickle
pickle.dump(LR, open( "save.p", "wb" ) )